"""The provenance chain. Every test here is a link that, if it broke silently,
would let a result that verified nothing be reported as evidence."""

from __future__ import annotations

import json
from collections import namedtuple
from dataclasses import replace

import pytest
from conftest import INIT_HASH, PRETRAIN_WHEEL, REPOP_SHA, REPOP_WHEEL

from gensyn_audit import kit as K
from gensyn_audit.errors import AuditError


def test_reads_a_published_kit(kit_dir):
    k = K.load_kit(str(kit_dir))
    assert k.kit_id == "pt-3f60ef8ff6af_rp-244c0791e378"
    assert k.repop_commit == REPOP_SHA
    assert {f.name for f in k.files} == {PRETRAIN_WHEEL, REPOP_WHEEL, "trajectory.json"}


def test_repop_is_installed_first(kit_dir):
    """Its wheel pins the torch the digests were produced under; installing
    pretrain first lets pip resolve a different one."""
    order = K.load_kit(str(kit_dir)).wheels_in_install_order("macosx")
    assert [f.name for f in order] == [REPOP_WHEEL, PRETRAIN_WHEEL]


def test_no_wheel_for_this_platform_is_an_error_not_a_fallback(kit_dir):
    with pytest.raises(AuditError, match="no repop wheel for linux"):
        K.load_kit(str(kit_dir)).wheels_in_install_order("linux")


def test_selects_native_pretrain_for_platform(kit_dir):
    kit = K.load_kit(str(kit_dir))
    original = next(f for f in kit.files if f.name == PRETRAIN_WHEEL)
    mac = replace(original, name="pretrain-0.1.0-cp311-cp311-macosx_14_0_arm64.whl")
    linux = replace(original, name="pretrain-0.1.0-cp311-cp311-linux_x86_64.whl")
    wrong_python = replace(original, name="pretrain-0.1.0-cp312-cp312-macosx_14_0_arm64.whl")
    kit = replace(
        kit, files=tuple(f for f in kit.files if f != original) + (mac, linux, wrong_python)
    )
    assert [f.name for f in kit.wheels_in_install_order("macosx")] == [REPOP_WHEEL, mac.name]


def test_missing_native_pretrain_fails_before_install(kit_dir):
    kit = K.load_kit(str(kit_dir))
    kit = replace(
        kit,
        files=tuple(
            replace(f, name="pretrain-0.1.0-cp311-cp311-linux_x86_64.whl")
            if f.name == PRETRAIN_WHEEL
            else f
            for f in kit.files
        ),
    )
    with pytest.raises(AuditError, match="one pretrain wheel"):
        kit.wheels_in_install_order("macosx")


def test_ambiguous_pretrain_wheels_are_rejected(kit_dir):
    kit = K.load_kit(str(kit_dir))
    original = next(f for f in kit.files if f.name == PRETRAIN_WHEEL)
    kit = replace(
        kit,
        files=kit.files
        + (replace(original, name="pretrain-0.1.0-cp311-cp311-macosx_14_0_arm64.whl"),),
    )
    with pytest.raises(AuditError, match="one pretrain wheel"):
        kit.wheels_in_install_order("macosx")


# ── link 1: digests, before install ──────────────────────────────────────────


def test_staging_verifies_every_digest(kit_dir, tmp_path):
    k = K.load_kit(str(kit_dir))
    staged = K.stage(k, tmp_path / "stage")
    assert set(staged) == {PRETRAIN_WHEEL, REPOP_WHEEL, "trajectory.json"}


def test_a_tampered_file_is_refused_and_deleted(kit_dir, tmp_path):
    (kit_dir / REPOP_WHEEL).write_bytes(b"something else entirely")
    k = K.load_kit(str(kit_dir))
    with pytest.raises(AuditError, match="does not match the digest"):
        K.stage(k, tmp_path / "stage")
    assert not (tmp_path / "stage" / REPOP_WHEEL).exists(), "must not leave bad bytes staged"


def test_a_wrong_cached_file_is_refetched_not_trusted(kit_dir, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / REPOP_WHEEL).write_bytes(b"stale truncated download")
    k = K.load_kit(str(kit_dir))
    K.stage(k, stage)
    assert (stage / REPOP_WHEEL).read_bytes() == b"repop wheel bytes"


def test_staging_is_idempotent(kit_dir, tmp_path):
    k = K.load_kit(str(kit_dir))
    stage = tmp_path / "stage"
    K.stage(k, stage)
    seen: list[str] = []
    K.stage(k, stage, on_progress=lambda entry, what: seen.append(what))
    assert set(seen) == {"cached"}, "a verified file must not be re-fetched"


# ── link 4: trajectory and wheels must pair ──────────────────────────────────


def test_trajectory_from_another_repop_build_is_refused(kit_dir):
    doc = json.loads((kit_dir / "trajectory.json").read_text())
    doc["repop_commit"] = "0" * 40
    (kit_dir / "trajectory.json").write_text(json.dumps(doc))
    k = K.load_kit(str(kit_dir))
    with pytest.raises(AuditError, match="different repop build"):
        K.load_trajectory(k)


def test_units_carry_their_published_cost(kit_dir):
    traj = K.load_trajectory(K.load_kit(str(kit_dir)))
    unit = traj.find("1b_repop_run3", "init")
    assert unit.is_init and unit.target_hash == INIT_HASH
    assert (unit.mps_wall_seconds, unit.mps_peak_rss_gb) == (29.7, 5.8)


def test_an_unknown_unit_names_what_the_trajectory_has(kit_dir):
    traj = K.load_trajectory(K.load_kit(str(kit_dir)))
    with pytest.raises(AuditError) as exc:
        traj.find("does_not_exist")
    assert "1b_repop_run3" in exc.value.hint


# ── link 5: the result must name the build the kit pins ──────────────────────


def test_a_result_from_another_build_is_not_evidence(kit_dir):
    k = K.load_kit(str(kit_dir))
    K.verify_result_provenance({"repop": {"commit": REPOP_SHA}}, k)
    with pytest.raises(AuditError, match="different repop build"):
        K.verify_result_provenance({"repop": {"commit": "b" * 40}}, k)
    with pytest.raises(AuditError, match="different repop build"):
        K.verify_result_provenance({}, k)


# ── schema hygiene ───────────────────────────────────────────────────────────


def test_a_future_kit_format_is_refused_not_guessed_at(kit_dir):
    doc = json.loads((kit_dir / "kit.json").read_text())
    doc["kit_format"] = 99
    (kit_dir / "kit.json").write_text(json.dumps(doc))
    with pytest.raises(AuditError, match="kit_format 99"):
        K.load_kit(str(kit_dir))


def test_a_short_commit_is_refused(kit_dir):
    doc = json.loads((kit_dir / "kit.json").read_text())
    doc["repop_commit"] = REPOP_SHA[:12]
    (kit_dir / "kit.json").write_text(json.dumps(doc))
    with pytest.raises(AuditError, match="not a full commit sha"):
        K.load_kit(str(kit_dir))


# ── the descriptor guards, checked before anything large is downloaded ───────


def _meta(**over) -> bytes:
    import json

    doc = {
        "step": 25700,
        "reduction_mode": "deterministic_allgather",
        "replicate_reduce_algo": "recursive_doubling",
        "clip_algo": "global",
    }
    doc.update(over)
    return (json.dumps(doc)).encode()


def _unit(uri="gs://b/ckpt/step_000025700"):
    return K.Unit(
        kind="interval",
        config_name="r",
        until_step=25701,
        state_hash="a" * 64,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri=uri,
        gcs_root="gs://b/shards",
        expect_hash="a" * 64,
        predecessor_step=25700,
    )


def test_an_unauditable_clipper_is_caught_before_the_download(monkeypatch):
    """The case that cost a real 19 GB download: audit_replay applies this guard
    itself, but only after loading the checkpoint. meta.json is 5 KB."""
    from gensyn_audit import doctor, gcs

    monkeypatch.setattr(gcs, "get", lambda uri, creds=None: _meta(clip_algo="adagc"))
    checks = doctor._check_descriptor(_unit())
    assert any(c.blocking and c.name.endswith("clip_algo") for c in checks)
    assert "predates the stateless deterministic global-norm clip" in checks[0].fix


def test_a_non_auditable_run_is_caught_too(monkeypatch):
    from gensyn_audit import doctor, gcs

    monkeypatch.setattr(gcs, "get", lambda uri, creds=None: _meta(reduction_mode="nccl"))
    checks = doctor._check_descriptor(_unit())
    assert any(c.blocking and "reduction_mode" in c.name for c in checks)


def test_an_auditable_checkpoint_passes(monkeypatch):
    from gensyn_audit import doctor, gcs

    monkeypatch.setattr(gcs, "get", lambda uri, creds=None: _meta())
    checks = doctor._check_descriptor(_unit())
    assert len(checks) == 1 and not checks[0].blocking


def test_packed_handoff_descriptor_is_deferred_until_unpack(monkeypatch):
    from gensyn_audit import doctor, gcs

    monkeypatch.setattr(gcs, "get", lambda *args, **kwargs: pytest.fail("unexpected fetch"))
    checks = doctor._check_descriptor(_unit("gs://b/handoff.safetensors"))
    assert checks[0].status == doctor.SKIP
    assert checks[0].value == "deferred until authenticated unpack"


def test_absent_keys_mean_the_default_which_is_the_wanted_one(monkeypatch):
    """A pre-fork meta omits keys that were introduced later; their defaults are
    exactly the values the audit requires, so absence must not read as failure."""
    from gensyn_audit import doctor, gcs

    monkeypatch.setattr(
        gcs,
        "get",
        lambda uri, creds=None: b'{"step": 1, "reduction_mode": "deterministic_allgather"}',
    )
    assert not any(c.blocking for c in doctor._check_descriptor(_unit()))


def test_an_unreadable_descriptor_warns_rather_than_blocks(monkeypatch):
    """This is an optimisation, not a gate — audit_replay re-checks regardless,
    so a transient read failure must not stop an otherwise fine audit."""
    from gensyn_audit import doctor, gcs
    from gensyn_audit.errors import AuditError

    def boom(uri, creds=None):
        raise AuditError("network down")

    monkeypatch.setattr(gcs, "get", boom)
    checks = doctor._check_descriptor(_unit())
    assert checks and not checks[0].blocking


def test_an_init_unit_has_no_descriptor_to_check():
    from gensyn_audit import doctor

    unit = K.Unit(
        kind="init",
        config_name="c",
        until_step=0,
        state_hash="a" * 64,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri=None,
        gcs_root=None,
        expect_hash=None,
    )
    assert doctor._check_descriptor(unit) == []


# ── an access gateway is not a document ──────────────────────────────────────


def test_a_login_page_is_not_reported_as_bad_json(monkeypatch):
    """Cloudflare Access answers with its login page, so the body parses as
    nothing. Reporting "not valid JSON" sends people to inspect a file that is
    fine; the problem is their credentials."""
    import io
    import urllib.request

    from gensyn_audit import kit as kitmod
    from gensyn_audit.errors import AuditError

    class _Resp(io.BytesIO):
        headers = type("H", (), {"get_content_type": staticmethod(lambda: "text/html")})()

        def geturl(self):
            return "https://sso.example.test/cdn-cgi/access/login/rec.test"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(b"<html>sign in</html>"))
    with pytest.raises(AuditError) as e:
        kitmod._read("https://rec.test/manifest.json")
    msg = str(e.value) + (e.value.hint or "")
    assert "web page" in msg
    assert "sso.example.test" in msg  # names the wall that was hit
    assert "AUDIT_HTTP_HEADERS" in msg  # and how to get past it
    assert "JSON" not in str(e.value)


def test_a_real_document_still_reads(monkeypatch):
    import io
    import urllib.request

    from gensyn_audit import kit as kitmod

    class _Resp(io.BytesIO):
        headers = type("H", (), {"get_content_type": staticmethod(lambda: "application/json")})()

        def geturl(self):
            return "https://rec.test/manifest.json"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(b'{"ok":1}'))
    assert kitmod._read("https://rec.test/manifest.json") == b'{"ok":1}'


# ── public_url_base ──────────────────────────────────────────────────────────


def _kit_doc(**extra):
    return {
        "kit_format": 1,
        "pretrain_commit": "f" * 40,
        "repop_commit": "e" * 40,
        "files": [{"name": "trajectory.json", "sha256": "a" * 64, "bytes": 1}],
        **extra,
    }


def _load(monkeypatch, doc, prefix="gs://b/audit-kit/k"):
    import json as _json

    from gensyn_audit import kit as kitmod

    monkeypatch.setattr(kitmod, "_read", lambda uri: _json.dumps(doc).encode())
    return kitmod.load_kit(prefix)


def test_a_gs_kit_is_read_from_its_public_mirror(monkeypatch):
    """A kit named by gs:// needs Google credentials, and a volunteer has none.
    The publisher writes the same bytes somewhere public and records it here."""
    k = _load(monkeypatch, _kit_doc(public_url_base="https://example.test/kit"))
    assert k.prefix == "gs://b/audit-kit/k"  # provenance: where it was named
    assert k.public_prefix == "https://example.test/kit"  # and where it is read


def test_without_a_mirror_the_prefix_is_used(monkeypatch):
    k = _load(monkeypatch, _kit_doc())
    assert k.public_url_base is None
    assert k.public_prefix == "gs://b/audit-kit/k"


def test_a_trailing_slash_does_not_double(monkeypatch):
    k = _load(monkeypatch, _kit_doc(public_url_base="https://example.test/kit/"))
    assert k.public_prefix == "https://example.test/kit"


def test_a_gs_mirror_is_refused(monkeypatch):
    """The field exists to name a location needing no credentials. A gs:// value
    would defeat that silently."""
    import pytest as _p

    from gensyn_audit.errors import AuditError

    with _p.raises(AuditError) as exc:
        _load(monkeypatch, _kit_doc(public_url_base="gs://other/kit"))
    assert "http(s)" in str(exc.value)


def test_staged_files_come_from_the_mirror(monkeypatch, tmp_path):
    """Parsing the field is not the point; reading from it is."""
    from gensyn_audit import kit as kitmod

    k = _load(monkeypatch, _kit_doc(public_url_base="https://example.test/kit"))
    seen = []
    monkeypatch.setattr(kitmod, "_read", lambda uri: seen.append(uri) or b"{}")
    with pytest.raises(AuditError):
        kitmod.load_trajectory(k)
    assert seen and seen[0].startswith("https://example.test/kit/")


# ── which interpreter builds the kit venv ────────────────────────────────────


def test_our_own_interpreter_counts_when_nothing_is_on_path(monkeypatch):
    """`uv tool install --python 3.11` leaves nothing called python3.11 on
    PATH and runs this tool on 3.11 anyway. Looking only at PATH failed the
    supported install (2026-09-11) and sent the reader off to install Python
    by hand, which is what uv was brought in to avoid."""
    from gensyn_audit import doctor

    monkeypatch.setattr(K.shutil, "which", lambda name: None)
    monkeypatch.setattr(K.sys, "version_info", _Version(3, 11, 13, "final", 0))
    monkeypatch.setattr(K.sys, "executable", "/tools/gensyn-audit/bin/python")

    assert K.find_base_interpreter() == "/tools/gensyn-audit/bin/python"
    assert K._base_interpreter() == "/tools/gensyn-audit/bin/python"
    monkeypatch.setattr(doctor, "_run", lambda *a, **k: _Ran("Python 3.11.13"))
    monkeypatch.setattr(doctor.sys, "executable", "/tools/gensyn-audit/bin/python")
    check = doctor._check_python()
    assert check.status == doctor.PASS and not check.blocking
    assert check.note == "this tool's own interpreter"


def test_path_python311_still_wins_over_ours(monkeypatch):
    monkeypatch.setattr(K.shutil, "which", lambda name: "/usr/bin/python3.11")
    monkeypatch.setattr(K.sys, "version_info", _Version(3, 12, 9, "final", 0))
    assert K.find_base_interpreter() == "/usr/bin/python3.11"


def test_no_3_11_anywhere_is_refused_and_names_the_uv_install(monkeypatch):
    from gensyn_audit import doctor

    monkeypatch.setattr(K.shutil, "which", lambda name: None)
    monkeypatch.setattr(K.sys, "version_info", _Version(3, 13, 0, "final", 0))
    assert K.find_base_interpreter() is None
    with pytest.raises(AuditError) as exc:
        K._base_interpreter()
    assert "uv tool install --python 3.11" in (exc.value.hint or "")

    check = doctor._check_python()
    assert check.status == doctor.FAIL and check.blocking
    assert "uv tool install --python 3.11" in (check.fix or "")
    assert "brew install python@3.11" not in (check.fix or "")


class _Ran:
    def __init__(self, out: str) -> None:
        self.stdout, self.stderr, self.returncode = out, "", 0


#: `sys.version_info` is indexed *and* read by attribute, so a bare tuple is
#: not a stand-in for it.
_Version = namedtuple("_Version", "major minor micro releaselevel serial")
