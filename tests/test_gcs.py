"""The stdlib GCS client.

It exists so a volunteer needs no Cloud SDK: no several-hundred-megabyte
install, no `gcloud auth` dance, and on macOS no Gatekeeper prompt about an
unnotarized `gcloud-crc32c` that deletes itself when dismissed the obvious way.
"""

from __future__ import annotations

import base64
import hashlib
import os

import pytest

from gensyn_audit import gcs
from gensyn_audit.errors import AuditError


def test_uris_parse():
    u = gcs.parse("gs://bucket/a/b/c.txt")
    assert (u.bucket, u.name) == ("bucket", "a/b/c.txt")
    assert gcs.parse("gs://bucket").name == ""
    with pytest.raises(AuditError, match="not a gs://"):
        gcs.parse("https://example.com/x")
    with pytest.raises(AuditError, match="no bucket"):
        gcs.parse("gs://")


def test_crc32c_matches_the_known_vector():
    """Castagnoli, not the CRC32 in zlib. `"123456789"` is the standard check
    value; getting the polynomial reflected wrong still produces stable-looking
    garbage, so this pins it."""
    h = gcs.Crc32c()
    h.update(b"123456789")
    assert h.digest() == (0xE3069283).to_bytes(4, "big")


def test_crc32c_is_incremental():
    whole, split = gcs.Crc32c(), gcs.Crc32c()
    whole.update(b"the quick brown fox")
    split.update(b"the quick ")
    split.update(b"brown fox")
    assert whole.digest() == split.digest()


# ── choosing a checksum ──────────────────────────────────────────────────────


def test_md5_is_preferred_over_crc32c():
    """Not arbitrary: hashlib runs ~930 MB/s against ~20 MB/s for the pure
    Python crc32c — 21 seconds versus 16 minutes over a 19 GB checkpoint."""
    both = {"md5": b"m" * 16, "crc32c": b"c" * 4}
    _, want, algo = gcs._hasher(both)
    assert algo == "md5" and want == both["md5"]


def test_crc32c_is_used_when_md5_is_absent():
    """Composite objects carry no md5, and correctness beats speed there."""
    _, want, algo = gcs._hasher({"crc32c": b"c" * 4})
    assert algo == "crc32c" and want == b"c" * 4


def test_no_checksum_at_all_is_not_a_crash():
    assert gcs._hasher({}) is None


def test_a_download_hashed_with_the_wrong_algorithm_would_be_caught():
    """The response headers determine which checksum verifies the download."""
    data = b"payload"
    good = {"md5": hashlib.md5(data).digest()}
    gcs._verify(data, good, "gs://b/o")  # must not raise
    with pytest.raises(AuditError, match="md5 mismatch|corrupted"):
        gcs._verify(data, {"md5": b"0" * 16}, "gs://b/o")


def test_header_parsing():
    class H:
        @staticmethod
        def get_all(_):
            crc = base64.b64encode(b"abcd").decode()
            md5 = base64.b64encode(b"x" * 16).decode()
            return [f"crc32c={crc},md5={md5}"]

    out = gcs._expected_hashes(H())
    assert out["crc32c"] == b"abcd"
    assert out["md5"] == b"x" * 16


def test_a_malformed_hash_header_is_ignored_not_fatal():
    class H:
        @staticmethod
        def get_all(_):
            return ["crc32c=!!!not-base64!!!", "md5=" + base64.b64encode(b"y" * 16).decode()]

    out = gcs._expected_hashes(H())
    assert out.get("md5") == b"y" * 16


# ── credentials ──────────────────────────────────────────────────────────────


def test_anonymous_is_a_first_class_outcome(tmp_path, monkeypatch):
    """A bucket an auditor is meant to read needs no credential, and demanding
    one would be the extra step this module exists to remove."""
    monkeypatch.delenv("AUDIT_GCS_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path))
    creds = gcs.credentials()
    assert creds.anonymous
    assert creds.token() is None


def test_an_explicit_token_wins(monkeypatch):
    monkeypatch.setenv("AUDIT_GCS_TOKEN", "ya29.test")
    creds = gcs.credentials()
    assert not creds.anonymous and creds.token() == "ya29.test"


def test_an_adc_file_is_used_without_shelling_out(tmp_path, monkeypatch):
    import json

    monkeypatch.delenv("AUDIT_GCS_TOKEN", raising=False)
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path))
    (tmp_path / "application_default_credentials.json").write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "c",
                "client_secret": "s",
                "refresh_token": "r",
            }
        )
    )
    creds = gcs.credentials()
    assert not creds.anonymous
    assert "ADC" in creds.source


def test_a_service_account_key_says_what_to_do_instead(tmp_path, monkeypatch):
    """Signing a JWT assertion is more crypto than belongs in a downloader.
    Better to say so than to half-implement it."""
    import json

    monkeypatch.delenv("AUDIT_GCS_TOKEN", raising=False)
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path))
    (tmp_path / "application_default_credentials.json").write_text(
        json.dumps({"type": "service_account", "client_email": "x@y.iam"})
    )
    with pytest.raises(AuditError) as exc:
        gcs.credentials()
    assert "AUDIT_GCS_TOKEN" in exc.value.hint


def test_a_403_while_anonymous_explains_both_ways_forward(monkeypatch):
    import urllib.error

    err = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
    made = gcs._explain(err, "gs://b/o", gcs.Credentials("anonymous"))
    assert "not publicly readable" in str(made)
    assert "gcloud auth application-default login" in made.hint
    assert "AUDIT_GCS_TOKEN" in made.hint


# ── the path a public artifact actually takes ────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("AUDIT_NETWORK_TESTS"), reason="hits the network; set AUDIT_NETWORK_TESTS=1"
)
def test_anonymous_read_of_a_public_bucket(tmp_path, monkeypatch):
    """The end state: published artifacts are public, so an auditor needs no
    credentials and no SDK. Verified against a bucket that is public today, so
    the claim does not wait on our own buckets being opened up.
    """
    monkeypatch.delenv("AUDIT_GCS_TOKEN", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path / "no-gcloud-here"))

    creds = gcs.credentials()
    assert creds.anonymous

    objects = gcs.list_prefix("gs://gcp-public-data-landsat/LC08/01/044/034", creds)
    assert objects, "anonymous listing must work"

    small = min((o for o in objects if o["size"] > 1000), key=lambda o: o["size"])
    dest = tmp_path / "f.bin"
    gcs.download(
        f"gs://gcp-public-data-landsat/{small['name']}",
        dest,
        creds,
        expected={k: small[k] for k in ("md5", "crc32c") if small[k]},
    )
    assert dest.stat().st_size == small["size"], "downloaded and checksum-verified"


# ── aggregate download progress ──────────────────────────────────────────────


def test_progress_counts_bytes_within_a_file_not_just_between_files(monkeypatch, tmp_path):
    """A checkpoint is a handful of multi-gigabyte shards. Reporting only on
    file boundaries leaves the bar motionless for minutes at a time, which on a
    19 GB transfer is indistinguishable from a stall."""
    objects = [
        {"name": "p/dcp/__0_0.distcp", "size": 100, "md5": None, "crc32c": None},
        {"name": "p/dcp/__1_0.distcp", "size": 100, "md5": None, "crc32c": None},
    ]
    monkeypatch.setattr(gcs, "list_prefix", lambda uri, creds=None: objects)

    def fake_download(uri, dest, creds=None, *, expected=None, on_progress=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x" * 100)
        for done in (25, 50, 75, 100):  # partial reports within the object
            if on_progress:
                on_progress(done, 100)
        return dest

    monkeypatch.setattr(gcs, "download", fake_download)
    monkeypatch.setattr(gcs, "credentials", lambda: gcs.Credentials("anonymous"))

    seen = []
    total = gcs.download_prefix(
        "gs://b/p", tmp_path, on_progress=lambda d, t, i, c, cur: seen.append((d, t))
    )
    assert total == 200
    fractions = [d / t for d, t in seen]
    assert fractions == sorted(fractions), "progress must never go backwards"
    assert fractions[-1] == 1.0
    # Between the two file boundaries (0.5) there must be intermediate readings.
    assert any(0 < f < 0.5 for f in fractions), "no within-file progress reported"


def test_cached_files_still_advance_the_bar(monkeypatch, tmp_path):
    """A resumed transfer skips what it already has; the bar must account for
    those bytes or it restarts at zero and looks like no progress was kept."""
    objects = [{"name": "p/a.bin", "size": 100, "md5": None, "crc32c": None}]
    (tmp_path / "a.bin").write_bytes(b"y" * 100)
    monkeypatch.setattr(gcs, "list_prefix", lambda uri, creds=None: objects)
    monkeypatch.setattr(gcs, "credentials", lambda: gcs.Credentials("anonymous"))

    def boom(*a, **k):
        raise AssertionError("must not re-download a complete file")

    monkeypatch.setattr(gcs, "download", boom)

    seen = []
    gcs.download_prefix(
        "gs://b/p", tmp_path, on_progress=lambda d, t, i, c, cur: seen.append(d / t)
    )
    assert seen == [1.0]


# ── checksum speed is a correctness-adjacent concern ─────────────────────────


def test_the_fast_and_pure_python_crc32c_agree():
    """The C extension is optional; verification must never depend on it being
    installable, only its speed does."""
    from gensyn_audit.gcs import _PyCrc32c

    for payload in (b"", b"123456789", bytes(range(256)) * 97):
        fast, slow = gcs.Crc32c(), _PyCrc32c()
        fast.update(payload)
        slow.update(payload)
        assert fast.digest() == slow.digest(), f"disagreed on {len(payload)} bytes"


def test_crc32c_matches_the_standard_check_value_either_way():
    from gensyn_audit.gcs import _PyCrc32c

    want = (0xE3069283).to_bytes(4, "big")
    for maker in (gcs.Crc32c, _PyCrc32c):
        h = maker()
        h.update(b"123456789")
        assert h.digest() == want


def test_composite_objects_force_the_crc32c_path():
    """The reason the C extension is a dependency rather than an extra: the
    large shards of a checkpoint are composite, GCS publishes no md5 for those,
    and they are ~19 of the 19.3 GB. In pure Python that is the bottleneck."""
    composite = {"crc32c": b"abcd"}  # no md5, as GCS reports for these
    _, _, algo = gcs._hasher(composite)
    assert algo == "crc32c"
