"""What a packed hand-off must SAY while it downloads.

An auditor watching `gensyn-audit run` cannot tell a slow multi-gigabyte
transfer from a hung process, so silence on the transport path is not cosmetic:
it is the one failure mode where the correct action (wait) and the wrong one
(kill it and start over) look identical. The packed branch used to print
nothing at all, not even the URI it was reading, while the directory branch
beside it drew a bar with bytes, a rate and an estimate.

These pin the reporting rather than the transfer. The bar itself only redraws
on a terminal, so what is asserted here is the part that holds everywhere: the
source is named before the bytes move, a progress callback reaches gcs, and the
size is stated when it lands.
"""

from __future__ import annotations

from gensyn_audit.kit import Unit

PACKED = "gs://bucket/handoffs/step_000000101/handoff.safetensors"


def _unit(uri: str) -> Unit:
    return Unit(
        kind="interval",
        config_name="c",
        until_step=101,
        state_hash="a" * 64,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri=uri,
        gcs_root=None,
        expect_hash=None,
    )


def _fake_download(calls: list[dict], *, payload: bytes = b"x" * 4096):
    """Stand in for gcs.download, recording how it was called.

    Drives the callback the way the real one does, in blocks, so a caller that
    passes a badly shaped callback fails here rather than in front of an
    auditor at two in the morning.
    """

    def download(uri, dest, creds=None, *, expected=None, on_progress=None):
        calls.append({"uri": uri, "on_progress": on_progress})
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        if on_progress is not None:
            half = len(payload) // 2
            on_progress(half, len(payload))
            on_progress(len(payload), len(payload))
        return dest

    return download


def test_a_packed_handoff_names_its_source_and_reports_progress(monkeypatch, tmp_path, capsys):
    from gensyn_audit import fetch, gcs

    calls: list[dict] = []
    monkeypatch.setattr(gcs, "download", _fake_download(calls))

    dest = tmp_path / "checkpoint" / "step_000000101"
    bundle = fetch.fetch_checkpoint(PACKED, dest, _unit(PACKED), expected_digest=None)

    assert bundle == dest / "handoff.safetensors"
    assert bundle.is_file()
    assert len(calls) == 1 and calls[0]["uri"] == PACKED
    assert calls[0]["on_progress"] is not None, (
        "the packed path handed gcs.download no progress callback, so a "
        "multi-gigabyte transfer prints nothing and reads as a hang"
    )

    out = capsys.readouterr().out
    assert PACKED in out, "the packed path did not name the object it was reading"
    from gensyn_audit.ui import human_bytes

    assert f"{human_bytes(bundle.stat().st_size)} fetched" in out, (
        f"no size reported when the bundle landed: {out!r}"
    )


def test_the_descriptor_fetch_stays_quiet(monkeypatch, tmp_path):
    """A few kilobytes of meta.json gets no bar.

    It already prints its own line when it lands, and a progress block for one
    small object is noise that trains people to ignore the block that matters.
    """
    from gensyn_audit import fetch, gcs

    calls: list[dict] = []
    monkeypatch.setattr(gcs, "download", _fake_download(calls, payload=b"{}"))

    fetch.fetch_descriptor("gs://bucket/forks/step_000000100", tmp_path / "descriptor")

    assert len(calls) == 1
    assert calls[0]["uri"].endswith("/meta.json")
    assert calls[0]["on_progress"] is None, (
        "the descriptor fetch drew a progress block for a kilobyte file"
    )
