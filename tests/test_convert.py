"""Driving the kit's own DCP <-> safetensors converter.

The conversion itself lives in the pretrain wheel, versioned with the sharded
layout it understands. What is tested here is the CLI's side: that it calls the
right direction at the right moment, refuses clearly when a kit cannot do it,
and never turns a good audit into a failure.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from gensyn_audit import convert
from gensyn_audit.errors import AuditError
from gensyn_audit.plan import Workdir


def _kit_venv(tmp_path: Path, script: str) -> Path:
    """A venv whose `pretrain-dcp-safetensors` is the given shell script."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    exe = venv / "bin" / convert.ENTRYPOINT
    exe.write_text(script)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return venv


def _checkpoint(tmp_path: Path) -> Path:
    from conftest import write_handoff

    return write_handoff(tmp_path / "handoff" / "step_000025701", step=25701)


def test_pack_writes_the_single_file_the_record_takes(tmp_path):
    venv = _kit_venv(tmp_path, '#!/bin/sh\nprintf tensors > "$3"\n')
    out = convert.pack(venv, _checkpoint(tmp_path), tmp_path / "out" / convert.BUNDLE_NAME)
    assert out.is_file() and out.read_text() == "tensors"


def test_pack_asks_for_verification(tmp_path):
    """--verify re-reads the file and compares every tensor. It runs once per
    audit against bytes the next auditor replays from."""
    seen = tmp_path / "argv"
    venv = _kit_venv(tmp_path, f'#!/bin/sh\necho "$@" > {seen}\nprintf x > "$3"\n')
    convert.pack(venv, _checkpoint(tmp_path), tmp_path / "b.safetensors")
    argv = seen.read_text().split()
    assert argv[0] == "to-safetensors"
    assert "--verify" in argv
    assert "--keys" not in argv, "a --keys pack is weight-only and cannot be resumed from"


def test_unpack_rejects_a_weight_only_bundle(tmp_path):
    """A bundle packed from a bare dcp/ dir restores no meta.json. Replaying
    from it diverges, and the divergence is indistinguishable from a real one."""
    venv = _kit_venv(tmp_path, '#!/bin/sh\nmkdir -p "$3/dcp"\n')  # no meta.json
    bundle = tmp_path / "handoff.safetensors"
    bundle.write_text("x")
    with pytest.raises(AuditError) as exc:
        convert.unpack(venv, bundle, tmp_path / "ckpt")
    assert "resume sidecar" in (exc.value.hint or "")


def test_unpack_clears_a_half_written_destination(tmp_path):
    """The converter refuses a destination that already holds files, so an
    interrupted run must not make the retry unstartable."""
    venv = _kit_venv(tmp_path, '#!/bin/sh\nmkdir -p "$3"\nprintf "{}" > "$3/meta.json"\n')
    bundle = tmp_path / "handoff.safetensors"
    bundle.write_text("x")
    dest = tmp_path / "ckpt"
    dest.mkdir()
    (dest / "leftover.distcp").write_text("from a killed run")
    convert.unpack(venv, bundle, dest)
    assert not (dest / "leftover.distcp").exists()


def test_a_kit_without_the_converter_says_so(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    for direction, call in (
        ("produce", lambda: convert.pack(venv, _checkpoint(tmp_path), tmp_path / "b")),
        ("start from", lambda: convert.unpack(venv, tmp_path / "b", tmp_path / "c")),
    ):
        (tmp_path / "b").write_text("x")
        with pytest.raises(AuditError) as exc:
            call()
        assert direction in str(exc.value)
        assert convert.ENTRYPOINT in str(exc.value)


def test_a_converter_failure_carries_its_own_output(tmp_path):
    venv = _kit_venv(tmp_path, '#!/bin/sh\necho "shard 7 is truncated" >&2\nexit 3\n')
    with pytest.raises(AuditError) as exc:
        convert.pack(venv, _checkpoint(tmp_path), tmp_path / "b.safetensors")
    assert "shard 7 is truncated" in (exc.value.hint or "")


def test_silent_success_is_not_trusted(tmp_path):
    """Exit 0 having written nothing would otherwise be submitted as an artifact."""
    venv = _kit_venv(tmp_path, "#!/bin/sh\nexit 0\n")
    with pytest.raises(AuditError) as exc:
        convert.pack(venv, _checkpoint(tmp_path), tmp_path / "b.safetensors")
    assert "wrote no" in str(exc.value)


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell stub")
def test_pack_names_the_directory_it_came_from(tmp_path):
    seen = tmp_path / "argv"
    venv = _kit_venv(tmp_path, f'#!/bin/sh\necho "$@" > {seen}\nprintf x > "$3"\n')
    src = _checkpoint(tmp_path)
    convert.pack(venv, src, tmp_path / "b.safetensors")
    assert str(src) in seen.read_text()


# ── the two call sites ───────────────────────────────────────────────────────


def test_the_receipt_describes_the_bundle_not_the_directory(tmp_path):
    """`artifactDigest` is declared at submit and the verifier checks the
    arriving bytes against it, so the receipt must describe the file that is
    actually uploaded, not the directory it was packed from."""
    from gensyn_audit import submit as submit_mod
    from gensyn_audit.plan import Workdir
    from gensyn_audit.progress import Progress
    from gensyn_audit.runner import RunState
    from gensyn_audit.upload import digest_file

    bundle = tmp_path / "handoff.safetensors"
    bundle.write_bytes(b"packed tensors")

    state = RunState(
        kit_id="pt-x_rp-y",
        kit_prefix="p",
        run="open-1b",
        unit_kind="interval",
        config_name="c",
        until_step=25701,
        audit_step=25700,
        expect_hash="a" * 64,
        device="mps",
        pid=1,
        argv=[],
        env_overlay={},
        started_at="2026-09-10T00:00:00+00:00",
        workdir=str(tmp_path),
    )
    prog = Progress()
    prog.state_hash = "a" * 64
    prog.match = True

    result = submit_mod.build(state, prog, Workdir(tmp_path), bundle=bundle)
    assert result.artifact["path"] == str(bundle)
    assert result.artifact["bytes"] == len(b"packed tensors")
    assert result.artifact["digest"] == digest_file(bundle)


def test_without_a_bundle_the_receipt_still_names_the_directory(tmp_path):
    """A kit with no converter still produces a checkpoint dir. The receipt
    should say what exists rather than claim an artifact that does not."""
    from gensyn_audit import submit as submit_mod
    from gensyn_audit.plan import Workdir
    from gensyn_audit.progress import Progress
    from gensyn_audit.runner import RunState

    wd = Workdir(tmp_path)
    wd.handoff.mkdir(parents=True, exist_ok=True)
    (wd.handoff / "meta.json").write_text("{}")

    state = RunState(
        kit_id="k",
        kit_prefix="p",
        run="r",
        unit_kind="interval",
        config_name="c",
        until_step=25701,
        audit_step=25700,
        expect_hash="a" * 64,
        device="mps",
        pid=1,
        argv=[],
        env_overlay={},
        started_at="2026-09-10T00:00:00+00:00",
        workdir=str(tmp_path),
    )
    result = submit_mod.build(state, Progress(), wd, bundle=None)
    assert result.artifact["path"] == str(wd.handoff)
    assert "digest" not in result.artifact


def test_a_pack_failure_does_not_fail_the_audit(tmp_path, monkeypatch, capsys):
    """The replay matched and the hash stands. What is lost is the artifact the
    next auditor would start from, not this auditor's result."""
    from gensyn_audit import cli
    from gensyn_audit.outcome import Outcome

    src = tmp_path / "handoff" / "step_000025701"
    src.mkdir(parents=True)

    class _Plan:
        venv = tmp_path / "venv"

        class workdir:
            handoff = tmp_path / "handoff"

    class _State:
        unit_kind = "interval"

    class _Prog:
        saved_checkpoint = None

    def fail_pack(*args, **kwargs):
        raise AuditError("converter blew up")

    monkeypatch.setattr(cli.convertmod, "pack", fail_pack)
    out = cli._pack_handoff(_Plan(), _State(), _Prog(), Outcome.MATCH)
    assert out is None
    captured = capsys.readouterr()
    assert "converter blew up" in captured.err + captured.out


def test_an_init_unit_packs_nothing(tmp_path):
    from gensyn_audit import cli
    from gensyn_audit.outcome import Outcome

    class _Plan:
        venv = tmp_path

        class workdir:
            handoff = tmp_path

    class _State:
        unit_kind = "init"

    class _Prog:
        saved_checkpoint = None

    assert cli._pack_handoff(_Plan(), _State(), _Prog(), Outcome.MATCH) is None


def test_a_published_predecessor_is_never_unpacked(tmp_path):
    """Position 1 fetches a real DCP directory from the bucket. Converting it
    would be wrong, and the branch must key on the record's word, not the shape
    of what arrived."""
    from gensyn_audit.record import Predecessor

    calls = []

    class _Ctx:
        predecessor = Predecessor(
            source="published checkpoint",
            uri="gs://b/ckpt/step_000025700/",
            digest=None,
            step=25700,
        )

    class _Plan:
        venv = tmp_path
        unit = type("U", (), {"predecessor_step": 25700})()

        class workdir:
            @staticmethod
            def checkpoint_dir(step):
                return tmp_path / f"step_{step:09d}"

    import gensyn_audit.convert as c

    original = c.unpack
    c.unpack = lambda *a, **k: calls.append(a)
    try:
        from gensyn_audit.verify import gate

        gate(checkpoint=tmp_path, pred=_Ctx.predecessor, venv=tmp_path, state_hashes=None)
    finally:
        c.unpack = original
    assert calls == [], "converted a published checkpoint"


def test_crowd_bundle_is_refused_before_unpacking(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gensyn_audit import cli
    from gensyn_audit.record import Predecessor

    def must_not_unpack(*args, **kwargs):
        pytest.fail("unverified bundle reached the converter")

    monkeypatch.setattr(cli.convertmod, "unpack", must_not_unpack)
    bundle = tmp_path / "handoff.safetensors"
    bundle.write_bytes(b"unverified")
    ctx = SimpleNamespace(
        predecessor=Predecessor(
            source="crowd-provided checkpoint", uri=str(bundle), digest=None, step=100
        )
    )
    from gensyn_audit.verify import gate

    with pytest.raises(AuditError, match="no usable artifact digest"):
        gate(
            checkpoint=bundle,
            pred=ctx.predecessor,
            venv=tmp_path,
            state_hashes=None,
            unpack_to=tmp_path / "restored",
        )
    from gensyn_audit.kit import KitFile

    ctx.predecessor = Predecessor(
        source="crowd-provided checkpoint",
        uri=str(bundle),
        digest=None,
        step=100,
        artifact_files=(KitFile(bundle.name, "0" * 64, bundle.stat().st_size),),
    )
    with pytest.raises(AuditError, match="not the file the record published"):
        gate(
            checkpoint=bundle,
            pred=ctx.predecessor,
            venv=tmp_path,
            state_hashes=None,
            unpack_to=tmp_path / "restored",
        )


def test_failed_pack_does_not_publish_or_replace_a_cached_bundle(tmp_path):
    venv = _kit_venv(tmp_path, '#!/bin/sh\nprintf partial > "$3"\nexit 1\n')
    target = tmp_path / "handoff.safetensors"
    target.write_bytes(b"previous complete bundle")
    with pytest.raises(AuditError):
        convert.pack(venv, _checkpoint(tmp_path), target)
    assert target.read_bytes() == b"previous complete bundle"
    assert not list(tmp_path.glob(".packing-*"))


def test_unpack_cannot_delete_its_own_source(tmp_path):
    venv = _kit_venv(tmp_path, "#!/bin/sh\nexit 0\n")
    bundle = tmp_path / "source" / "handoff.safetensors"
    bundle.parent.mkdir()
    bundle.write_bytes(b"original")
    with pytest.raises(AuditError, match="must not contain the source"):
        convert.unpack(venv, bundle, bundle.parent)
    assert bundle.read_bytes() == b"original"


def test_verified_directory_is_used_for_replay(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gensyn_audit import cli
    from gensyn_audit.verify import Verdict

    restored = tmp_path / "restored"
    monkeypatch.setattr(
        cli.verifymod, "gate", lambda **kw: Verdict("crowd", "verified", "verified", restored)
    )
    plan = SimpleNamespace(
        checkpoint_path=lambda: tmp_path / "handoff.safetensors",
        venv=tmp_path,
        device="cpu",
        workdir=Workdir(tmp_path),
        checkpoint=None,
    )
    cli._verify_predecessor(plan, SimpleNamespace(), SimpleNamespace(is_crowd_provided=False))
    assert plan.checkpoint == restored


def test_fetch_single_bundle_does_not_treat_transport_digest_as_state_hash(tmp_path, monkeypatch):
    from gensyn_audit import fetch, gcs

    uri = "gs://bucket/handoff.safetensors"

    # Same signature as the real gcs.download: the packed path passes a
    # progress callback, and a stub that refuses one turns a reporting change
    # into a TypeError here instead of covering what this test is about.
    def download(source, dest, creds=None, *, expected=None, on_progress=None):
        assert source == uri
        dest.write_bytes(b"packed checkpoint")

    monkeypatch.setattr(gcs, "download", download)
    dest = tmp_path / "download"
    bundle = fetch.fetch_checkpoint(uri, dest, None, expected_digest="d" * 64)
    assert bundle == dest / "handoff.safetensors"
    assert bundle.read_bytes() == b"packed checkpoint"
    # The later shared gate authenticates the bundle, including on local reuse.
    assert fetch.fetch_checkpoint(bundle.as_uri(), bundle, None, expected_digest="d" * 64) == bundle


# ── the predecessor gate is answered before the download ─────────────────────


def _ctx(kind: str, files=()):
    from gensyn_audit.record import Predecessor, StepContext

    return StepContext(
        run="open-1b",
        step=25700,
        committed_hash="a" * 64,
        predecessor=Predecessor(
            source=kind, uri="gs://b/ckpt/", digest=None, step=25700, artifact_files=tuple(files)
        ),
        descriptor_uri=None,
        phase="main",
        microbatches=288,
        gcs_root=None,
    )


def test_a_digestless_anchor_is_trusted_without_warning():
    from gensyn_audit.doctor import PASS, _check_predecessor_digest

    (check,) = _check_predecessor_digest(_ctx("published checkpoint"))
    assert check.status is PASS
    assert check.value == "Trusted Gensyn anchor"


def test_a_crowd_handoff_has_no_override():
    """Trusting Gensyn anchors does not authorize another auditor's bytes."""
    from gensyn_audit.doctor import FAIL, _check_predecessor_digest

    (check,) = _check_predecessor_digest(_ctx("crowd-provided checkpoint"))
    assert check.status is FAIL
    assert "hand-off" in (check.note or "")


def test_published_digests_pass():
    from gensyn_audit.doctor import PASS, _check_predecessor_digest
    from gensyn_audit.kit import KitFile

    files = [KitFile(name="meta.json", sha256="b" * 64, bytes=10)]
    (check,) = _check_predecessor_digest(_ctx("published checkpoint", files))
    assert check.status is PASS


@pytest.mark.parametrize("digest", ["abc", "g" * 64, "a" * 63, "a" * 65])
def test_malformed_bundle_digest_fails_preflight_and_never_unpacks(tmp_path, monkeypatch, digest):
    from dataclasses import replace

    from gensyn_audit import doctor, verify

    ctx = _ctx("crowd-provided checkpoint")
    ctx = replace(ctx, predecessor=replace(ctx.predecessor, digest=digest))
    assert doctor._check_predecessor_digest(ctx)[0].status == doctor.FAIL
    bundle = tmp_path / convert.BUNDLE_NAME
    bundle.write_bytes(b"bundle")
    monkeypatch.setattr(convert, "unpack", lambda *a: pytest.fail("unpacked invalid digest"))
    with pytest.raises(AuditError, match="64 hexadecimal"):
        verify.gate(checkpoint=bundle, pred=ctx.predecessor, venv=tmp_path, state_hashes=None)


def test_unknown_predecessor_cannot_use_anchor_override():
    from gensyn_audit import doctor

    assert doctor._check_predecessor_digest(_ctx("unknown"))[0].status == doctor.FAIL


def test_an_init_unit_has_no_predecessor_to_check():
    from gensyn_audit.doctor import _check_predecessor_digest

    assert _check_predecessor_digest(None) == []


@pytest.mark.parametrize("tamper", [False, True])
@pytest.mark.parametrize("cached_directory", [False, True])
def test_record_bundle_digest_is_checked_before_unpack(
    tmp_path, monkeypatch, tamper, cached_directory
):
    from types import SimpleNamespace

    from conftest import write_handoff

    from gensyn_audit import cli, doctor, record, upload, verify
    from gensyn_audit.steps import StepRef

    bundle = tmp_path / "handoff.safetensors"
    bundle.write_bytes(b"uploaded bundle")
    ctx = record._parse_step_context(
        "run",
        100,
        {
            "step": 100,
            "committed": "c" * 64,
            "predecessor": {
                "kind": "crowd-provided checkpoint",
                "uri": bundle.as_uri(),
                "digest": upload.digest_file(bundle),
                "check": "loss-checked",
            },
        },
    )
    assert not ctx.predecessor.artifact_files and ctx.predecessor.step is None
    assert doctor._check_predecessor_digest(ctx)[0].status == doctor.PASS
    restored = write_handoff(tmp_path / "restored", step=100)
    calls = []

    def unpack(*args):
        assert not tamper, "tampered bytes reached the converter"
        calls.append("unpack")
        return restored

    monkeypatch.setattr(convert, "unpack", unpack)
    monkeypatch.setattr(
        verify, "check_state_hash", lambda *a, **kw: {"step": 100, "dp_world_size": 1}
    )
    log = tmp_path / "hashes.jsonl"
    log.write_text(
        json.dumps({"step": 99, "state_hash": "a" * 64})
        + "\n"
        + json.dumps({"step": 100, "state_hash": "b" * 64})
        + "\n"
    )
    checkpoint = bundle.parent if cached_directory else bundle
    plan = SimpleNamespace(
        checkpoint_path=lambda: checkpoint,
        venv=tmp_path,
        device="cpu",
        unit=SimpleNamespace(predecessor_step=StepRef.from_audit(100).predecessor_log),
        workdir=Workdir(tmp_path / "receiver"),
    )
    args = SimpleNamespace(state_hashes=str(log))
    if tamper:
        bundle.write_bytes(b"altered bundle")
        with pytest.raises(AuditError, match="BLAKE2b-256 mismatch"):
            cli._verify_predecessor(plan, args, ctx.predecessor)
        assert not calls
    else:
        assert cli._verify_predecessor(plan, args, ctx.predecessor).artifact == "verified"
        assert calls == ["unpack"]
        assert plan.checkpoint == restored
