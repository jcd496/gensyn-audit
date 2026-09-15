"""Detached execution and reuse of locally produced predecessors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
from test_interval_end_to_end import COMMITTED, RUN, _run

from gensyn_audit import cli, handoffs
from gensyn_audit.mock import MockRecord


def _wait(predicate, *, timeout: float = 90.0, what: str = "") -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what or predicate}")


def _state(world) -> dict:
    return json.loads((world["workdir"] / "run.json").read_text())


def _gone(pid: int) -> bool:
    """Has this process exited? The supervisor is a child of the test process,
    which -- unlike the real parent, which exits at once -- lives on and has to
    reap it, or a finished supervisor sits as a zombie that `kill -0` still
    answers for. In production the parent is gone and launchd/init reaps."""
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return True
    except ChildProcessError:
        pass
    return not cli.runner.is_running(pid)


# ── --detach finishes the audit on its own ───────────────────────────────────


def test_detach_replays_reports_and_submits_without_a_rerun(world, capsys):
    code = _run(world, "--detach", "--claim", "clm_detached")
    out = capsys.readouterr().out
    assert code == 0
    assert "started detached" in out
    assert "Nothing needs re-running" in out
    assert "Re-run this exact command" not in out

    state = _state(world)
    assert state["detached"] is True
    assert state["supervisor_pid"] > 0
    supervisor = state["supervisor_pid"]

    _wait(lambda: _gone(supervisor), what="the detached run to finish")

    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["outcome"] == "match"
    assert receipt["reproduced_hash"] == COMMITTED
    # The verdict about the starting checkpoint travelled from the parent, which
    # ran the gate, to the child, which replayed and reported.
    assert receipt["predecessor"]["statement"] == "Trusted Gensyn anchor"
    assert "checkpoint" not in receipt["predecessor"], "a local path leaked into the receipt"
    assert "200" in MockRecord(world["fixture"]).state.accepted, "the child did not submit"

    screen = (world["workdir"] / "gensyn-audit.log").read_text()
    assert "MATCH" in screen and "recorded" in screen
    final = _state(world)
    assert final["supervisor_pid"] == supervisor, "the child lost track of who launched it"
    assert final["detached"] is True
    assert final["pid"] > 0, "the child never recorded the replay it launched"

    capsys.readouterr()
    assert cli.main(["status", "--workdir", str(world["workdir"]), "--no-color"]) == 0
    shown = capsys.readouterr().out
    assert "MATCH" in shown
    assert "receipt" in shown
    assert "Re-run the same" not in shown, "status told a finished audit to re-run"


def _run_without_workdir(world, *extra) -> int:
    return cli.main(
        [
            "run",
            "--kit",
            str(world["kit"]),
            "--step",
            "200",
            "--run",
            RUN,
            "--record",
            f"mock://{world['fixture']}",
            "--skip-doctor",
            "--no-color",
            *extra,
        ]
    )


def _finished_in(root: Path) -> None:
    """The detached run that `run.json` in `root` describes ran to a receipt in
    `root`, and never lost track of itself along the way."""
    state = json.loads((root / "run.json").read_text())
    supervisor = state["supervisor_pid"]
    assert supervisor > 0
    _wait(lambda: _gone(supervisor), what="the detached run to finish")
    receipt = json.loads((root / "result.json").read_text())
    assert receipt["outcome"] == "match"
    final = json.loads((root / "run.json").read_text())
    assert final["supervisor_pid"] == supervisor
    assert final["pid"] > 0
    assert final["workdir"] == str(root)


def test_a_relative_workdir_lands_the_child_where_the_parent_staged(world, monkeypatch, capsys):
    """The child uses the absolute workdir prepared by its parent."""
    monkeypatch.chdir(world["workdir"].parent)
    code = _run_without_workdir(world, "--workdir", "rel-wd", "--detach", "--claim", "clm_rel")
    assert code == 0
    root = world["workdir"].parent / "rel-wd"
    assert f"--workdir {root}" in capsys.readouterr().out, "status hint must name the real dir"
    _finished_in(root)
    assert not (root / "rel-wd").exists(), "the child nested a second workdir inside the first"
    assert "200" in MockRecord(world["fixture"]).state.accepted


def test_an_omitted_workdir_means_the_same_directory_to_parent_and_child(world, monkeypatch):
    monkeypatch.chdir(world["workdir"].parent)
    assert _run_without_workdir(world, "--detach", "--claim", "clm_default") == 0
    roots = [p for p in world["workdir"].parent.glob("audit-*") if p.is_dir()]
    assert len(roots) == 1, roots
    _finished_in(roots[0])
    assert not list(roots[0].glob("audit-*")), "the child nested a second workdir inside the first"
    assert "200" in MockRecord(world["fixture"]).state.accepted


def test_the_claim_reaches_the_child_without_touching_its_argv_or_the_log(world, monkeypatch):
    """argv is public to every user on the machine for the whole replay, and
    the supervisor echoes its command into `gensyn-audit.log`. The token goes
    down a pipe instead, and the child still submits with it."""
    import subprocess

    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    class Recording(real_popen):
        def __init__(self, command, *a, **k):
            spawned.append([str(t) for t in command])
            super().__init__(command, *a, **k)

    monkeypatch.setattr(cli.runner.subprocess, "Popen", Recording)
    assert _run(world, "--detach", "--claim", "clm_secret") == 0
    children = [c for c in spawned if "gensyn_audit.cli" in c]
    assert len(children) == 1, spawned
    assert "--claim" not in children[0]
    assert "clm_secret" not in " ".join(children[0])

    supervisor = _state(world)["supervisor_pid"]
    _wait(lambda: _gone(supervisor), what="the detached run to finish")
    screen = (world["workdir"] / "gensyn-audit.log").read_text()
    assert "clm_secret" not in screen
    assert "gensyn_audit.cli" in screen, "the header still says what was run"
    record = MockRecord(world["fixture"]).state
    assert "200" in record.accepted, "the child never received the claim"
    assert "clm_secret" in record.used_claims


def test_the_parent_cannot_overwrite_the_replay_pid_the_child_records(world, tmp_path, monkeypatch):
    """The child records its replay pid only after the parent's state write."""
    venv_bin = next((tmp_path / "cache").rglob("bin"))
    (venv_bin / "pretrain-audit-replay").write_text("#!/bin/sh\nsleep 60\n")
    (venv_bin / "pretrain-audit-replay").chmod(0o755)

    real = cli.runner._state_for

    def slow(*a, **k):
        time.sleep(2.0)  # a free-running child launches its replay well inside this
        return real(*a, **k)

    monkeypatch.setattr(cli.runner, "_state_for", slow)
    assert _run(world, "--detach") == 0
    supervisor = _state(world)["supervisor_pid"]
    _wait(lambda: _state(world)["pid"] > 0, timeout=20, what="the child to record its replay")
    state = _state(world)
    assert state["supervisor_pid"] == supervisor
    replay = state["pid"]
    assert cli.runner.is_running(replay)

    assert cli.main(["stop", "--workdir", str(world["workdir"]), "--yes", "--no-color"]) == 0
    _wait(lambda: _gone(supervisor), timeout=10, what="supervisor exit")
    _wait(lambda: not cli.runner.is_running(replay), timeout=10, what="replay exit")


def test_the_supervised_child_does_not_stage_the_predecessor_again(world, monkeypatch):
    """The child reuses the predecessor verdict saved by its parent."""
    calls = []
    monkeypatch.setattr(cli, "_stage_predecessor", lambda *a, **k: calls.append("stage"))
    monkeypatch.setattr(cli, "_verify_predecessor", lambda *a, **k: calls.append("gate"))

    (world["workdir"]).mkdir(parents=True, exist_ok=True)
    pred = json.loads(world["fixture"].read_text())["steps"]["200"]["predecessor"]["uri"]
    (world["workdir"] / "predecessor.json").write_text(
        json.dumps(
            {
                "verdict": {
                    "predecessor_provenance": "gensyn-anchor",
                    "artifact_integrity": "verified",
                    "tensor_state_commitment": "skipped-anchor",
                    "statement": "Trusted Gensyn anchor",
                },
                "checkpoint": pred,
            }
        )
    )
    assert _run(world, "--supervised") == 0
    assert calls == []
    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["predecessor"]["statement"] == "Trusted Gensyn anchor"


def test_a_second_run_while_the_detached_one_is_alive_is_refused(world, monkeypatch, capsys):
    (world["workdir"]).mkdir(parents=True, exist_ok=True)
    from gensyn_audit.plan import Workdir
    from gensyn_audit.runner import RunState, save_state

    save_state(
        Workdir(world["workdir"]),
        RunState(
            kit_id="k",
            kit_prefix="p",
            run=RUN,
            unit_kind="interval",
            config_name="c",
            until_step=201,
            audit_step=200,
            expect_hash=COMMITTED,
            device="mps",
            pid=0,
            argv=[],
            env_overlay={},
            started_at="2026-09-12T00:00:00+00:00",
            workdir=str(world["workdir"]),
            detached=True,
            supervisor_pid=os.getpid() + 1,
        ),
    )
    monkeypatch.setattr(cli.runner, "is_running", lambda pid: pid == os.getpid() + 1)
    assert _run(world) == cli.EXIT_ERROR
    err = capsys.readouterr().err
    assert "detached audit is already in progress" in err
    assert "nothing needs re-running" in err


def test_stop_ends_the_detached_run_before_it_can_submit_a_cancellation(world, tmp_path, capsys):
    venv_bin = next((tmp_path / "cache").rglob("bin"))
    (venv_bin / "pretrain-audit-replay").write_text("#!/bin/sh\nsleep 60\n")
    (venv_bin / "pretrain-audit-replay").chmod(0o755)

    assert _run(world, "--detach", "--claim", "clm_stopped") == 0
    _wait(lambda: _state(world)["pid"] > 0, what="the child to launch the replay")
    state = _state(world)
    supervisor, replay = state["supervisor_pid"], state["pid"]
    assert cli.runner.is_running(supervisor) and cli.runner.is_running(replay)

    capsys.readouterr()
    assert cli.main(["stop", "--workdir", str(world["workdir"]), "--yes", "--no-color"]) == 0
    assert "stopped" in capsys.readouterr().out
    _wait(lambda: _gone(supervisor), timeout=10, what="supervisor exit")
    _wait(lambda: not cli.runner.is_running(replay), timeout=10, what="replay exit")
    time.sleep(0.5)
    assert not (world["workdir"] / "result.json").exists(), (
        "the supervisor outlived its replay long enough to report the kill as a result"
    )
    assert MockRecord(world["fixture"]).state.used_claims == []


def test_status_reports_a_detached_run_that_died_before_reporting(world, capsys):
    assert _run(world) == 0
    state = _state(world)
    state.update(detached=True, supervisor_pid=999_999, pid=999_998)
    (world["workdir"] / "run.json").write_text(json.dumps(state))
    (world["workdir"] / "result.json").unlink()
    capsys.readouterr()

    assert cli.main(["status", "--workdir", str(world["workdir"]), "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "ended before it could report" in out
    assert "Re-run the same `gensyn-audit run`" in out


# ── a hand-off this machine packed is not downloaded again ───────────────────


def _crowd_ctx(digest: str, *, step: int = 200):
    from gensyn_audit.record import Predecessor, StepContext

    return StepContext(
        run=RUN,
        step=step + 1,
        committed_hash=COMMITTED,
        predecessor=Predecessor(
            source="crowd-provided checkpoint",
            uri="gs://record/handoffs/step-200.safetensors",
            digest=digest,
            step=step,
        ),
        descriptor_uri=None,
        phase="main",
        microbatches=288,
        gcs_root="gs://b/shards",
    )


def _plan_for(tmp_path: Path, ctx):
    from types import SimpleNamespace

    from gensyn_audit import plan as plan_mod

    unit = SimpleNamespace(checkpoint_uri=ctx.predecessor.uri, predecessor_step=200, is_init=False)
    return plan_mod.Plan(
        unit=unit, kit=None, workdir=plan_mod.Workdir(tmp_path / "wd"), venv=tmp_path / "venv"
    )


def test_a_bundle_this_machine_packed_is_used_in_place(tmp_path, monkeypatch, capsys):
    bundle = tmp_path / "audit-200" / "handoff" / "handoff.safetensors"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"the bytes that were uploaded")
    digest = hashlib.blake2b(bundle.read_bytes(), digest_size=32).hexdigest()
    handoffs.remember(bundle, digest, step=200, run=RUN)

    def no_download(*a, **k):
        raise AssertionError("fetched a hand-off this machine already holds")

    monkeypatch.setattr(cli.fetch, "fetch_checkpoint", no_download)
    ctx = _crowd_ctx(digest)
    plan = _plan_for(tmp_path, ctx)
    cli._stage_predecessor(plan, argparse.Namespace(refetch=False, predecessor_uri=None), ctx)

    assert plan.checkpoint == bundle
    out = capsys.readouterr().out
    assert "this machine packed this hand-off" in out
    assert "Not downloaded again" in out
    assert "digest" in out, "must say the gate still runs on it"


@pytest.mark.parametrize(
    "why", ["refetch", "override", "unknown digest", "file changed", "file gone"]
)
def test_the_download_still_happens_when_the_local_copy_cannot_be_trusted(
    tmp_path, monkeypatch, why
):
    bundle = tmp_path / "handoff.safetensors"
    bundle.write_bytes(b"packed")
    digest = hashlib.blake2b(b"packed", digest_size=32).hexdigest()
    handoffs.remember(bundle, digest, step=200)
    fetched = []
    monkeypatch.setattr(
        cli.fetch, "fetch_checkpoint", lambda uri, dest, *a, **k: fetched.append(uri) or dest
    )

    args = argparse.Namespace(
        refetch=why == "refetch", predecessor_uri="gs://elsewhere" if why == "override" else None
    )
    if why == "unknown digest":
        digest = "0" * 64
    elif why == "file changed":
        bundle.write_bytes(b"packed, then edited")
    elif why == "file gone":
        bundle.unlink()

    ctx = _crowd_ctx(digest)
    cli._stage_predecessor(_plan_for(tmp_path, ctx), args, ctx)
    assert fetched == [ctx.predecessor.uri]


def test_the_index_is_written_when_the_bundle_is_digested_for_upload(world, monkeypatch):
    """`upload` is where the digest is first known, so that is where the
    bundle is remembered -- before the transfer, so a retry tomorrow still
    finds it."""
    from gensyn_audit import upload as uploadmod
    from gensyn_audit.record import UploadTicket

    bundle_file = world["workdir"] / "handoff.safetensors"
    bundle_file.parent.mkdir(parents=True, exist_ok=True)
    bundle_file.write_bytes(b"tensors")
    digest = hashlib.blake2b(b"tensors", digest_size=32).hexdigest()
    monkeypatch.setattr(uploadmod, "send", lambda *a, **k: None)
    monkeypatch.setattr(uploadmod, "send_sidecar", lambda *a, **k: None)
    monkeypatch.setattr(uploadmod, "sidecar", lambda *a, **k: {})
    monkeypatch.setattr(cli, "_record_run_id", lambda *a: "run-id")

    class _Rec:
        is_mock = False

        @staticmethod
        def upload_ticket(*a, **k):
            return UploadTicket(signed_url="https://x/b", sidecar_url="https://x/s")

    class _Plan:
        class workdir:
            handoff = world["workdir"]
            root = world["workdir"]

    class _State:
        run, until_step, audit_step, claim = RUN, 201, 200, "clm_x"

    class _Result:
        artifact = {"path": str(bundle_file)}

    assert handoffs.find(digest) is None
    cli._do_upload(_Plan(), _State(), _Rec(), None, _Result())
    known = handoffs.find(digest)
    assert known is not None and known.path == bundle_file and known.step == 200


# ── a downloaded bundle is consumed once the gate has passed ─────────────────


def _crowd_verdict(plan, bundle: Path, digest: str):
    """What `verify.gate` returns for a packed crowd hand-off: the unpacked
    directory to replay, and the bundle it was held to the record's digest."""
    from gensyn_audit.verify import Verdict

    verified = plan.workdir.verified_predecessor
    verified.mkdir(parents=True, exist_ok=True)
    (verified / "meta.json").write_text('{"step": 100}')
    plan.checkpoint = verified
    return Verdict("crowd", "verified", "verified", verified, bundle=bundle, bundle_digest=digest)


def _gate_a_bundle_at(monkeypatch, bundle: Path):
    """Stand in for the download and the gate around one bundle, wherever it
    sits, so the run exercises what `cmd_run` does with the verdict."""
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"eighteen gigabytes of tensors")
    digest = hashlib.blake2b(bundle.read_bytes(), digest_size=32).hexdigest()

    def stage(plan, args, ctx):
        plan.checkpoint = bundle
        return False

    monkeypatch.setattr(cli, "_stage_predecessor", stage)
    monkeypatch.setattr(
        cli, "_verify_predecessor", lambda plan, *a, **k: _crowd_verdict(plan, bundle, digest)
    )
    return digest


def test_a_downloaded_bundle_is_gone_after_the_gate(world, monkeypatch, capsys):
    """Once the gate has unpacked and verified it, nothing reads the bundle
    again -- and at 18 GB it was the largest thing in the workdir with no
    remaining consumer."""
    bundle = world["workdir"] / "checkpoint" / "step_000000100" / "handoff.safetensors"
    digest = _gate_a_bundle_at(monkeypatch, bundle)

    assert _run(world) == 0
    out = capsys.readouterr().out
    assert not bundle.exists(), "the downloaded bundle outlived the gate"
    assert (world["workdir"] / "verified-predecessor" / "meta.json").is_file()
    assert "reclaimed" in out and "nothing reads it again" in out

    saved = json.loads((world["workdir"] / "predecessor.json").read_text())
    assert saved["bundle_digest"] == digest, "the restart path needs the verified digest"
    assert saved["checkpoint"] == str(world["workdir"] / "verified-predecessor")
    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert "bundle_digest" not in receipt["predecessor"]


def test_a_reused_bundle_survives_the_gate(world, monkeypatch, tmp_path):
    """The previous workdir's `handoff/handoff.safetensors` is that audit's
    upload source and this machine's reuse source for the next step. It was
    never downloaded here, and it is not this workdir's to delete."""
    bundle = tmp_path / "audit-100" / "handoff" / "handoff.safetensors"
    _gate_a_bundle_at(monkeypatch, bundle)

    assert _run(world) == 0
    assert bundle.is_file(), "a bundle outside this workdir's checkpoint/ was deleted"
    assert bundle.read_bytes() == b"eighteen gigabytes of tensors"


def test_the_bundle_stays_when_the_verdict_could_not_be_saved(world, monkeypatch):
    """Without `predecessor.json` a `--restart` has nothing to say the
    verified directory is trustworthy, so the bundle is the only thing it
    could re-verify. Deleting it then would force the download."""
    bundle = world["workdir"] / "checkpoint" / "step_000000100" / "handoff.safetensors"
    _gate_a_bundle_at(monkeypatch, bundle)
    monkeypatch.setattr(cli, "_remember_predecessor", lambda *a, **k: False)

    assert _run(world) == 0
    assert bundle.is_file()


def test_only_a_regular_file_under_checkpoint_is_consumed(tmp_path):
    from gensyn_audit.plan import Workdir

    wd = Workdir(tmp_path / "wd")
    inside = wd.checkpoint_dir(100) / "handoff.safetensors"
    inside.parent.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere" / "handoff.safetensors"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"someone else's bytes")

    assert wd.consume_predecessor_bundle(None) == 0
    assert wd.consume_predecessor_bundle(inside) == 0, "a missing bundle is not an error"
    assert wd.consume_predecessor_bundle(elsewhere) == 0 and elsewhere.is_file()

    # A link inside checkpoint/ to bytes outside it must not reach them.
    inside.symlink_to(elsewhere)
    assert wd.consume_predecessor_bundle(inside) == 0 and elsewhere.is_file()
    inside.unlink()

    inside.write_bytes(b"downloaded here")
    assert wd.consume_predecessor_bundle(inside) == len(b"downloaded here")
    assert not inside.exists()
    assert wd.checkpoint_dir(100).is_dir(), "only the bundle goes, not the directory"


@pytest.mark.parametrize("link_at", ["checkpoint", "step"])
def test_consumption_does_not_follow_checkpoint_directory_symlinks(tmp_path, link_at):
    from gensyn_audit.plan import Workdir

    wd = Workdir(tmp_path / "wd")
    external = tmp_path / "shared"
    external.mkdir()
    bundle = external / "handoff.safetensors"
    bundle.write_bytes(b"externally owned bundle")
    link = wd.checkpoint_root if link_at == "checkpoint" else wd.checkpoint_dir(100)
    link.parent.mkdir(parents=True)
    link.symlink_to(external, target_is_directory=True)

    assert wd.consume_predecessor_bundle(link / bundle.name) == 0
    assert bundle.read_bytes() == b"externally owned bundle"


# ── --restart replays from the verified directory, --refetch downloads ───────


def _remember_verified(plan, digest: str, *, verdict: str = "verified") -> Path:
    verified = plan.workdir.verified_predecessor
    verified.mkdir(parents=True, exist_ok=True)
    (verified / "meta.json").write_text('{"step": 200}')
    plan.workdir.predecessor_record.write_text(
        json.dumps(
            {
                "verdict": {
                    "predecessor_provenance": "crowd",
                    "artifact_integrity": "verified",
                    "tensor_state_commitment": verdict,
                    "statement": "Crowd hand-off; verified.",
                },
                "checkpoint": str(verified),
                "bundle_digest": digest,
            }
        )
    )
    return verified


def test_restart_replays_from_the_verified_predecessor_without_a_download(
    tmp_path, monkeypatch, capsys
):
    """The bundle is gone -- consumed when the gate passed -- and the record
    beside the verified directory says which bytes it came from. Fetching
    18 GB again to re-derive a directory already on disk is the gap."""
    digest = hashlib.blake2b(b"consumed", digest_size=32).hexdigest()
    ctx = _crowd_ctx(digest)
    plan = _plan_for(tmp_path, ctx)
    verified = _remember_verified(plan, digest)

    def no_download(*a, **k):
        raise AssertionError("--restart re-downloaded a hand-off this workdir already verified")

    monkeypatch.setattr(cli.fetch, "fetch_checkpoint", no_download)
    args = argparse.Namespace(restart=True, refetch=False, predecessor_uri=None)
    assert cli._stage_predecessor(plan, args, ctx) is True, "the gate already ran on it"
    assert plan.checkpoint == verified
    out = capsys.readouterr().out
    assert "already unpacked and verified" in out
    assert "--refetch" in out, "must say how to force a fresh download"


def test_failed_reunpack_invalidates_the_restart_verdict(tmp_path, monkeypatch):
    from gensyn_audit.errors import AuditError

    bundle = tmp_path / "handoff.safetensors"
    bundle.write_bytes(b"authenticated bundle")
    digest = hashlib.blake2b(bundle.read_bytes(), digest_size=32).hexdigest()
    ctx = _crowd_ctx(digest)
    plan = _plan_for(tmp_path, ctx)
    verified = _remember_verified(plan, digest)
    plan.checkpoint = bundle
    exe = cli.convertmod.converter(plan.venv)
    exe.parent.mkdir(parents=True)
    exe.touch()

    def fail_conversion(*args):
        # convert.unpack has already cleared and recreated the destination.
        assert not (verified / "meta.json").exists()
        (verified / "partial").write_bytes(b"incomplete")
        raise AuditError("unpacking failed")

    monkeypatch.setattr(cli.convertmod, "_run", fail_conversion)
    with pytest.raises(AuditError, match="unpacking failed"):
        cli._verify_predecessor(plan, argparse.Namespace(), ctx.predecessor)
    assert (verified / "partial").is_file()
    assert not plan.workdir.predecessor_record.exists()

    fetched = []
    monkeypatch.setattr(
        cli.fetch, "fetch_checkpoint", lambda uri, dest, *a, **k: fetched.append(uri) or dest
    )
    args = argparse.Namespace(restart=True, refetch=False, predecessor_uri=None)
    assert cli._stage_predecessor(plan, args, ctx) is False
    assert fetched == [ctx.predecessor.uri]


def test_verification_stops_if_the_old_verdict_cannot_be_invalidated(tmp_path, monkeypatch):
    from gensyn_audit.errors import AuditError

    ctx = _crowd_ctx("a" * 64)
    plan = _plan_for(tmp_path, ctx)
    verified = _remember_verified(plan, ctx.predecessor.digest)
    original_unlink = Path.unlink

    def refuse_verdict_unlink(path, *args, **kwargs):
        if path == plan.workdir.predecessor_record:
            raise PermissionError("read-only verdict")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_verdict_unlink)
    monkeypatch.setattr(cli.verifymod, "gate", lambda **kw: pytest.fail("gate must not run"))
    with pytest.raises(AuditError, match="could not invalidate the predecessor verdict"):
        cli._verify_predecessor(plan, argparse.Namespace(), ctx.predecessor)
    assert plan.workdir.predecessor_record.is_file()
    assert (verified / "meta.json").is_file()


def test_restart_honors_an_explicit_predecessor_uri(tmp_path, monkeypatch):
    ctx = _crowd_ctx("a" * 64)
    plan = _plan_for(tmp_path, ctx)
    _remember_verified(plan, ctx.predecessor.digest)
    override = "gs://another/hand-off.safetensors"
    plan.unit.checkpoint_uri = override
    fetched = []
    monkeypatch.setattr(
        cli.fetch, "fetch_checkpoint", lambda uri, dest, *a, **k: fetched.append(uri) or dest
    )

    args = argparse.Namespace(restart=True, refetch=False, predecessor_uri=override)
    assert cli._stage_predecessor(plan, args, ctx) is False
    assert fetched == [override]


def test_refetch_downloads_even_when_the_verified_predecessor_is_present(tmp_path, monkeypatch):
    digest = hashlib.blake2b(b"consumed", digest_size=32).hexdigest()
    ctx = _crowd_ctx(digest)
    plan = _plan_for(tmp_path, ctx)
    _remember_verified(plan, digest)
    fetched = []
    monkeypatch.setattr(
        cli.fetch, "fetch_checkpoint", lambda uri, dest, *a, **k: fetched.append(uri) or dest
    )

    args = argparse.Namespace(restart=True, refetch=True, predecessor_uri=None)
    assert cli._stage_predecessor(plan, args, ctx) is False, "a fresh download must be gated"
    assert fetched == [ctx.predecessor.uri]


@pytest.mark.parametrize(
    "why", ["no record", "record names other bytes", "directory gone", "gate never passed"]
)
def test_restart_stages_afresh_when_the_verified_directory_cannot_be_trusted(
    tmp_path, monkeypatch, why
):
    digest = hashlib.blake2b(b"consumed", digest_size=32).hexdigest()
    ctx = _crowd_ctx(digest)
    plan = _plan_for(tmp_path, ctx)
    if why == "record names other bytes":
        # The record now publishes a different hand-off for this step; the
        # directory here came from the old one.
        _remember_verified(plan, "0" * 64)
    elif why == "directory gone":
        import shutil

        shutil.rmtree(_remember_verified(plan, digest))
    elif why == "gate never passed":
        _remember_verified(plan, digest, verdict="skipped-anchor")
    fetched = []
    monkeypatch.setattr(
        cli.fetch, "fetch_checkpoint", lambda uri, dest, *a, **k: fetched.append(uri) or dest
    )

    args = argparse.Namespace(restart=True, refetch=False, predecessor_uri=None)
    assert cli._stage_predecessor(plan, args, ctx) is False
    assert fetched == [ctx.predecessor.uri]


def test_a_restarted_run_reports_the_verdict_it_replayed_from(world, monkeypatch, capsys):
    """End to end: the first run consumes the bundle; `--restart` neither
    downloads nor re-gates, and the receipt still says what the gate
    established."""
    real_stage = cli._stage_predecessor
    bundle = world["workdir"] / "checkpoint" / "step_000000100" / "handoff.safetensors"
    digest = _gate_a_bundle_at(monkeypatch, bundle)
    assert _run(world) == 0
    assert not bundle.exists()

    # The record names this step's predecessor as the crowd hand-off whose
    # digest the gate confirmed.
    doc = json.loads(world["fixture"].read_text())
    doc["steps"]["200"]["predecessor"] = {
        "source": "crowd-provided checkpoint",
        "step": 100,
        "uri": "gs://record/handoffs/step-100.safetensors",
        "digest": digest,
    }
    world["fixture"].write_text(json.dumps(doc))
    monkeypatch.setattr(cli, "_stage_predecessor", real_stage)

    def no_download(*a, **k):
        raise AssertionError("--restart re-downloaded the consumed bundle")

    def no_gate(*a, **k):
        raise AssertionError("--restart re-ran the gate on a directory it already verified")

    monkeypatch.setattr(cli.fetch, "fetch_checkpoint", no_download)
    monkeypatch.setattr(cli, "_verify_predecessor", no_gate)
    capsys.readouterr()
    assert _run(world, "--restart") == 0
    out = capsys.readouterr().out
    assert "already unpacked and verified" in out
    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["predecessor"]["tensor_state_commitment"] == "verified"
    assert "checkpoint" not in receipt["predecessor"]


# ── the report path does not redo the predecessor ────────────────────────────


def test_reporting_a_finished_replay_does_not_refetch_or_regate(world, monkeypatch, capsys):
    assert _run(world) == 0
    # Reporting later, from a workdir whose receipt is gone: the verdict the
    # gate saved is what the new receipt must carry.
    (world["workdir"] / "result.json").unlink()
    assert (world["workdir"] / "predecessor.json").is_file()

    calls = []
    monkeypatch.setattr(cli, "_stage_predecessor", lambda *a, **k: calls.append("stage"))
    monkeypatch.setattr(cli, "_verify_predecessor", lambda *a, **k: calls.append("gate"))
    assert _run(world, "--claim", "clm_report") == 0
    out = capsys.readouterr().out

    assert calls == []
    assert "already holds a finished replay" in out
    assert "Trusted Gensyn anchor" in out, "the report lost what the gate established"
    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["predecessor"]["statement"] == "Trusted Gensyn anchor"
    assert "checkpoint" not in receipt["predecessor"]


# ── the next step is spelled out ─────────────────────────────────────────────


def test_the_next_step_command_advances_the_step_and_asks_for_a_new_claim():
    args = argparse.Namespace(
        step=102,
        workdir="~/audits/open-1b-102",
        raw_argv=[
            "run",
            "--manifest",
            "m.json",
            "--step",
            "102",
            "--claim",
            "clm_old",
            "--workdir",
            "~/audits/open-1b-102",
            "--restart",
            "--supervised",
            "--extra",
            "--verbose-replay",
        ],
    )
    text = cli._next_step_command(args)
    assert "--step 103" in text
    assert "--claim '<claim for step 103>'" in text
    assert "clm_old" not in text
    assert "open-1b-103" in text
    assert "--restart" not in text
    assert "--supervised" not in text
    assert "--detach" in text, "a detached audit suggests a detached next step"
    assert text.index("--detach") < text.index("--extra"), (
        "flags after --extra would be handed to audit_replay"
    )


@pytest.mark.parametrize(
    "step, name, expected",
    [
        (102, "open-1b-102", "open-1b-103"),
        (12, "step-12-2026", "step-13-2026"),
        # The step's digits inside a longer number are not the step.
        (103, "open-1b-1030", "open-1b-1030-104"),
        (1, "run-101", "run-101-2"),
        (7, "audits", "audits-8"),
    ],
)
def test_the_next_step_workdir_advances_only_the_step_in_the_name(step, name, expected):
    args = argparse.Namespace(
        step=step,
        workdir=f"/a/{name}",
        raw_argv=["run", "--step", str(step), "--workdir", f"/a/{name}"],
    )
    text = cli._next_step_command(args)
    assert f"--workdir /a/{expected}" in text, text


def test_the_supervised_marker_precedes_any_replay_passthrough():
    args = argparse.Namespace(raw_argv=["run", "--step", "1", "--detach", "--extra", "--x"])
    argv = cli._supervised_argv(args, Path("/abs/wd"))
    assert argv[:4] == ["run", "--supervised", "--workdir", "/abs/wd"]
    assert "--detach" not in argv
    assert argv[-2:] == ["--extra", "--x"]


def test_the_child_gets_the_resolved_workdir_and_no_claim():
    """Whatever `--workdir` was spelled as, the child gets the directory the
    parent staged, absolute, so its own cwd cannot change what it means. The
    claim does not travel in argv at all (`runner.supervise` pipes it)."""
    args = argparse.Namespace(
        raw_argv=[
            "run",
            "--workdir",
            "wd",
            "--claim",
            "clm_1",
            "--step",
            "1",
            "--claim=clm_2",
            "--workdir=elsewhere",
            "--detach",
            "--extra",
            "--workdir",
            "kept-for-the-replay",
        ]
    )
    argv = cli._supervised_argv(args, Path("/abs/wd"))
    assert argv == [
        "run",
        "--supervised",
        "--workdir",
        "/abs/wd",
        "--step",
        "1",
        "--extra",
        "--workdir",
        "kept-for-the-replay",
    ]
    assert "clm_1" not in " ".join(argv) and "clm_2" not in " ".join(argv)


def test_the_supervisor_log_header_never_carries_the_claim():
    shown = cli.runner.redact_argv(["x", "--claim", "clm_1", "--claim=clm_2", "--step", "1"])
    assert shown == ["x", "--claim", "<redacted>", "--claim=<redacted>", "--step", "1"]


def test_the_hand_off_is_read_only_from_a_pipe():
    import io

    read = cli.runner.read_handoff
    assert read(io.StringIO('{"claim": "clm_x"}')) == {"claim": "clm_x"}
    assert read(io.StringIO("")) == {}
    assert read(io.StringIO("not json")) == {}
    assert read(io.StringIO("[1]")) == {}

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    # A terminal has no parent on the other end; blocking on it would hang a
    # hand-driven `--supervised` run. Same for pytest's captured stdin.
    assert read(Tty('{"claim": "x"}')) == {}
    assert read() == {}


def test_no_next_step_for_an_init_unit():
    assert cli._next_step_command(argparse.Namespace(step=None, raw_argv=["run"])) is None
