"""One interval audit, end to end, through the CLI's own entry point.

The replay is stubbed — a real one is ~18 GB of checkpoint and half a day — but
everything around it is the production path: kit resolution and its provenance
gate, step context from the record, the predecessor digest check, outcome
classification, the loss gate, submission, and the resumable upload.

The stub speaks audit_replay's dialect exactly: the same logging format, the
same result object with `repop` / `device` / `rank0_losses`, and the same
`--loss-log` file written alongside.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from gensyn_audit import cli
from gensyn_audit.mock import MockRecord

RUN = "20260703-171943-4e85cd3"
COMMITTED = "49a35d246590678b78088163da5c4600591e203d72beae008419ec5e479bdc2a"
PRED_DIGEST = "16b3d656df3b2c9ac9dd3dacdfad3a0132f7e17c0c175c0f369eae519d776e30"
REPOP_SHA = "244c0791e378a180dd6b29bbf8f2e244b2bea806"
TRUE_CE, TRUE_ZL = 4.8151623042000000, 0.0234567890123456

_BUILD_INFO = json.dumps({"commit": REPOP_SHA, "backends": ["cpu", "metal"], "cuda_arch_list": ""})

_REPLAY_STUB = """#!/bin/sh
# Stands in for pretrain-audit-replay. Emits exactly what it emits.
LOSS_LOG=""
EXPECT=""
UNTIL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --loss-log) LOSS_LOG="$2"; shift 2;;
    --expect-hash) EXPECT="$2"; shift 2;;
    --until-step) UNTIL="$2"; shift 2;;
    *) shift;;
  esac
done
echo "2026-09-07 10:00:00,000 INFO pretrain.audit :: replaying steps 100 -> $UNTIL"
echo "2026-09-07 10:00:01,000 INFO pretrain.audit :: MEMLOG[post-step] host_RSS=21.4GB mps_alloc=12.0GB mps_driver=19.8GB"
PRODUCED="__PRODUCED__"
[ -n "$LOSS_LOG" ] && cat > "$LOSS_LOG" <<LOSSES
{"records": [{"step": $UNTIL, "consumed_tokens": 4096, "loss_ce": __CE__, "loss_zloss": __ZL__}]}
LOSSES
echo "2026-09-07 10:00:02,000 INFO pretrain.audit :: state_hash=`echo $PRODUCED | cut -c1-16` expected=`echo $EXPECT | cut -c1-16` MATCH=true"
H="__HANDOFF__/step_$UNTIL"
mkdir -p "$H/dcp"
head -c 100000 /dev/zero > "$H/dcp/__0_0.distcp"
echo "$PRODUCED" > "$H/state_hash.txt"
echo "{\"step\": $UNTIL, \"dp_world_size\": 1, \"chained_hash\": \"$PRODUCED\"}" > "$H/meta.json"
echo "{}" > "$H/global_stream.json"
echo rng > "$H/rng.rank_0.pt"
echo grads > "$H/gradients.safetensors"
echo chain0 > "$H/batch_hasher.rank_0.bin"
: > "$H/_COMPLETE"
cat <<RESULT
{
  "step": $UNTIL,
  "consumed_tokens": 4096,
  "state_hash": "$PRODUCED",
  "repop": {"commit": "__REPOP__", "backends": ["cpu", "metal"], "cuda_arch_list": ""},
  "device": "mps",
  "rank0_losses": [{"step": $UNTIL, "consumed_tokens": 4096, "loss_ce": __CE__, "loss_zloss": __ZL__}],
  "saved_checkpoint": "__HANDOFF__/step_$UNTIL",
  "expected": "$EXPECT",
  "match": true
}
RESULT
"""


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A kit, a pre-provisioned venv with stubs, a predecessor, and a record."""
    monkeypatch.setenv("AUDIT_CACHE", str(tmp_path / "cache"))

    kit_dir = tmp_path / "kit"
    kit_dir.mkdir()
    wheels = {
        f"repop-0.1.5+g{REPOP_SHA[:12]}-cp311-cp311-macosx_14_0_arm64.whl": b"r",
        "pretrain-0.1.0-py3-none-any.whl": b"p",
    }
    for name, blob in wheels.items():
        (kit_dir / name).write_bytes(blob)
    traj = {
        "trajectory_format": 1,
        "name": "t",
        "repop_commit": REPOP_SHA,
        "pretrain_commit": "a" * 40,
        "notes": "",
        "units": [
            {
                "kind": "init",
                "config_name": "1b_repop_run3",
                "until_step": 0,
                "state_hash": "e" * 64,
                "devices_verified": ["mps"],
            }
        ],
    }
    tb = (json.dumps(traj) + "\n").encode()
    (kit_dir / "trajectory.json").write_bytes(tb)
    (kit_dir / "kit.json").write_text(
        json.dumps(
            {
                "kit_format": 1,
                "pretrain_commit": "a" * 40,
                "repop_commit": REPOP_SHA,
                "files": [
                    {"name": n, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
                    for n, b in wheels.items()
                ]
                + [
                    {
                        "name": "trajectory.json",
                        "sha256": hashlib.sha256(tb).hexdigest(),
                        "bytes": len(tb),
                    }
                ],
            }
        )
    )

    # Pre-provision the venv so `provision` short-circuits instead of pip-installing.
    kit_id = f"pt-{'a' * 12}_rp-{REPOP_SHA[:12]}"
    venv_bin = tmp_path / "cache" / "kits" / kit_id / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text(
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then\n'
        f"  case \"$2\" in *build_info*) echo '{_BUILD_INFO}';; "
        f"*python_version*) echo 3.11.6;; esac\nfi\n"
    )
    (venv_bin / "python").chmod(0o755)

    workdir = tmp_path / "wd"
    handoff = workdir / "handoff"
    (venv_bin / "pretrain-audit-replay").write_text(
        _REPLAY_STUB.replace("__PRODUCED__", COMMITTED)
        .replace("__REPOP__", REPOP_SHA)
        .replace("__CE__", repr(TRUE_CE))
        .replace("__ZL__", repr(TRUE_ZL))
        .replace("__HANDOFF__", str(handoff))
    )
    (venv_bin / "pretrain-audit-replay").chmod(0o755)

    # A local predecessor whose bytes carry the committed digest.
    pred = tmp_path / "pred" / "step_000000100"
    pred.mkdir(parents=True)
    (pred / "state_hash.txt").write_text(PRED_DIGEST + "\n")

    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "run": RUN,
                "gcs_root": None,
                "steps": {
                    "200": {
                        "committed_hash": COMMITTED,
                        "predecessor": {
                            "source": "published",
                            "step": 100,
                            "uri": str(pred),
                            "digest": PRED_DIGEST,
                            # What the record publishes about the anchor's BYTES.
                            # The committed state hash above is not this: it is a
                            # fact about the training state and survives a rewrite.
                            "artifactFiles": _artifact_files(pred),
                        },
                    }
                },
                "withheld_losses": {"200": {"loss_ce": TRUE_CE, "loss_zloss": TRUE_ZL}},
            }
        )
    )
    return {"kit": kit_dir, "fixture": fixture, "workdir": workdir}


def _artifact_files(root: Path) -> list[dict]:
    return [
        {
            "name": str(p.relative_to(root)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def _republish(world, root: Path) -> None:
    """Re-derive the record's digests after a test edits the anchor."""
    doc = json.loads(world["fixture"].read_text())
    doc["steps"]["200"]["predecessor"]["artifactFiles"] = _artifact_files(root)
    world["fixture"].write_text(json.dumps(doc))


def _run(world, *extra) -> int:
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
            "--workdir",
            str(world["workdir"]),
            "--skip-doctor",
            "--no-color",
            *extra,
        ]
    )


def test_a_matching_interval_is_recorded_without_partial_upload(world, capsys):
    code = _run(world, "--claim", "clm_real")
    captured = capsys.readouterr()
    out = captured.out + captured.err  # warnings go to stderr

    assert code == 0
    assert "MATCH" in out
    assert "pending verification" in out
    assert "receipt" in out
    assert "mock://receipt/" in out

    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["outcome"] == "match"
    assert receipt["reproduced_hash"] == COMMITTED
    assert receipt["repop_commit"] == REPOP_SHA
    assert receipt["losses"][-1]["loss_ce"] == TRUE_CE
    # The receipt must say a mock was involved. It is not a real audit record.
    assert receipt["mocked"]["record"] == "mock"

    # audit_replay writes a DCP checkpoint, not the safetensors hand-off the
    # record wants, so there is nothing to upload yet. That must be reported as
    # a missing hand-off — never as a failed audit.
    assert "relay does not advance" in out
    assert "pretrain-dcp-safetensors" in out
    assert "gradients.safetensors" in out, "the recipient's missing input must be named"
    assert "recorded either way" in out, "must not read as a failed audit"
    assert receipt["outcome"] == "match", "the audit itself still succeeded"


def test_digestless_anchor_runs_without_a_flag_and_records_the_assumption(world):
    doc = json.loads(world["fixture"].read_text())
    pred = doc["steps"]["200"]["predecessor"]
    pred.pop("artifactFiles")
    pred["digest"] = None
    world["fixture"].write_text(json.dumps(doc))

    assert _run(world) == 0
    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["outcome"] == "match"
    assert receipt["predecessor"]["artifact_integrity"] == "unverified"
    assert receipt["predecessor"]["tensor_state_commitment"] == "skipped-anchor"


def test_the_predecessor_digest_is_checked_before_anything_runs(world, capsys):
    pred = json.loads(world["fixture"].read_text())["steps"]["200"]["predecessor"]["uri"]
    Path(pred, "state_hash.txt").write_text("f" * 64 + "\n")

    code = _run(world, "--claim", "clm_bad_pred")
    out = capsys.readouterr().out + capsys.readouterr().err

    assert code == cli.EXIT_ERROR
    assert not (world["workdir"] / "result.json").exists(), "must not replay at all"
    del out


def test_a_wrong_loss_is_refused_even_though_the_hash_matched(world, capsys):
    """The gate is the losses, not the hash: the hash is public."""
    doc = json.loads(world["fixture"].read_text())
    doc["withheld_losses"]["200"]["loss_ce"] = TRUE_CE + 0.5
    world["fixture"].write_text(json.dumps(doc))

    code = _run(world, "--claim", "clm_wrong_loss")
    out = capsys.readouterr().out

    assert code == 0, "the replay itself still matched"
    assert "no public change" in out
    assert "does not match the cluster's value" in out
    assert MockRecord(world["fixture"]).state.accepted == {}


def test_without_a_claim_nothing_is_submitted(world, capsys):
    assert _run(world) == 0
    out = capsys.readouterr().out
    assert "Nothing was submitted" in out
    assert "no claim token" in out


def test_the_mock_banner_is_always_shown(world, capsys):
    _run(world, "--claim", "clm_banner")
    assert "MOCK RECORD" in capsys.readouterr().out


# ── resuming, and deadlines ──────────────────────────────────────────────────


def test_rerunning_after_a_finished_replay_reports_it_instead_of_replaying(world, capsys):
    """The bug this test exists for: re-running was starting a second 12-hour
    replay instead of submitting the finished one."""
    assert _run(world) == 0
    marker = (world["workdir"] / "audit.log").read_text().count("started")
    bundle = world["workdir"] / "handoff" / "handoff.safetensors"
    bundle.write_bytes(b"cached bundle")

    assert _run(world, "--claim", "clm_resume") == 0
    assert bundle.read_bytes() == b"cached bundle"
    out = capsys.readouterr().out

    assert "already holds a finished replay" in out
    assert (world["workdir"] / "audit.log").read_text().count("started") == marker, (
        "the replay must not have run a second time"
    )
    assert "recorded" in out, "the resumed run must submit"


@pytest.mark.parametrize("detached", [False, True])
def test_restart_clears_cached_bundle_before_launch(world, monkeypatch, detached):
    from types import SimpleNamespace

    assert _run(world) == 0
    bundle = world["workdir"] / "handoff" / "handoff.safetensors"
    bundle.write_bytes(b"stale bundle")
    calls = []
    original = cli.runner.run_foreground

    def launch(*args, **kwargs):
        assert not bundle.exists(), "fresh replay must not inherit a cached bundle"
        calls.append(True)
        return SimpleNamespace(pid=0, supervisor_pid=123) if detached else original(*args, **kwargs)

    monkeypatch.setattr(cli.runner, "supervise" if detached else "run_foreground", launch)
    assert _run(world, "--restart", *(["--detach"] if detached else [])) == 0
    assert calls == [True]


def test_restart_forces_a_fresh_replay(world):
    _run(world)
    before = (world["workdir"] / "audit.log").read_text().count("started")
    _run(world, "--restart")
    assert (world["workdir"] / "audit.log").read_text().count("started") == before + 1


def test_a_workdir_with_no_verdict_is_not_treated_as_finished(world):
    """A replay that died without a digest should start over, not be reported."""
    _run(world)
    log = world["workdir"] / "audit.log"
    log.write_text("2026-09-07 10:00:00,000 INFO pretrain.audit :: replaying steps 100 -> 200\n")
    assert _run(world) == 0
    assert "MATCH" in log.read_text() or "match" in log.read_text()


def test_a_running_replay_is_not_replaced(world, monkeypatch):
    _run(world)
    from gensyn_audit import runner as R

    monkeypatch.setattr(R, "is_running", lambda pid: True)
    assert _run(world) == cli.EXIT_ERROR


def test_a_deadline_stops_the_replay_and_reports_timeout(world, tmp_path, capsys):
    """`timeout` was an outcome the tool could never actually produce."""
    venv_bin = next((tmp_path / "cache").rglob("bin"))
    (venv_bin / "pretrain-audit-replay").write_text("#!/bin/sh\nsleep 30\n")
    (venv_bin / "pretrain-audit-replay").chmod(0o755)

    code = _run(world, "--timeout", "1s")
    out = capsys.readouterr().out

    assert code == 2, "a timeout is a finished-but-negative outcome, not a tool error"
    assert "TIMEOUT" in out
    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["outcome"] == "timeout"


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("90", 90.0),
        ("90s", 90.0),
        ("30m", 1800.0),
        ("18h", 64800.0),
        ("1.5h", 5400.0),
    ],
)
def test_durations_parse(text, seconds):
    from gensyn_audit.runner import parse_duration

    assert parse_duration(text) == seconds


def test_a_bad_duration_says_what_it_wanted():
    from gensyn_audit.errors import AuditError
    from gensyn_audit.runner import parse_duration

    with pytest.raises(AuditError) as exc:
        parse_duration("half a day")
    assert "18h" in exc.value.hint, "the error must show the accepted forms"


# ── preserve workdir evidence when not starting a replay ─────────────────────


def _spill(world) -> Path:
    """~18 GB of offload spill, in miniature."""
    d = world["workdir"] / "scratch" / "master"
    d.mkdir(parents=True, exist_ok=True)
    (d / "master_000.pt").write_bytes(b"weights")
    return d / "master_000.pt"


def test_reporting_a_finished_replay_keeps_its_scratch(world, capsys):
    assert _run(world) == 0
    spill = _spill(world)

    assert _run(world, "--claim", "clm_keep_scratch") == 0
    assert "already holds a finished replay" in capsys.readouterr().out
    assert spill.is_file(), "the report path deleted the evidence it was reporting"


def test_a_running_replay_keeps_its_scratch(world, monkeypatch):
    """The spill belongs to a live process; deleting it is not a cleanup."""
    _run(world)
    spill = _spill(world)

    from gensyn_audit import runner as R

    monkeypatch.setattr(R, "is_running", lambda pid: True)
    assert _run(world) == cli.EXIT_ERROR
    assert spill.is_file(), "cleared the live replay's offload spill before refusing"


def test_restart_does_clear_the_scratch(world):
    """The control: an actual replay still starts from a clean spill, which is
    what the clear was for -- a leftover would otherwise fail its own preflight."""
    assert _run(world) == 0
    spill = _spill(world)

    assert _run(world, "--restart") == 0
    assert not spill.exists()


def test_the_report_path_does_not_run_the_replay_sized_preflight(world, monkeypatch):
    """Keeping the spill must not then fail the preflight on it.

    The headroom checks size the machine for a replay. On the report path there
    is no replay to size for, and running them would fail the report on the
    very bytes it exists to preserve.
    """
    assert _run(world) == 0
    _spill(world)

    seen = []

    def spy(plan, **kw):
        seen.append(kw.get("for_replay"))
        return []  # all-pass, so this asserts the flag and nothing else

    monkeypatch.setattr(cli.doctor, "run_checks", spy)
    # _run passes --skip-doctor; drop it so the preflight is actually reached.
    code = cli.main(
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
            "--workdir",
            str(world["workdir"]),
            "--no-color",
        ]
    )
    assert code == 0
    assert seen == [False]


def test_a_real_replay_still_gets_the_headroom_checks(world, monkeypatch):
    seen = []

    def spy(plan, **kw):
        seen.append(kw.get("for_replay"))
        return []

    monkeypatch.setattr(cli.doctor, "run_checks", spy)
    code = cli.main(
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
            "--workdir",
            str(world["workdir"]),
            "--no-color",
        ]
    )
    assert code == 0
    assert seen == [True]


# ── recover the runtime of a detached audit ──────────────────────────────────


def _as_detached(world, *, started: str, finished_mtime: float) -> None:
    """Reshape a finished foreground run into what `--detach` leaves behind."""
    state = json.loads((world["workdir"] / "run.json").read_text())
    state["detached"] = True
    state["started_at"] = started
    state["finished_at"] = None
    state["exit_code"] = None
    state["pid"] = 999_999  # long gone
    (world["workdir"] / "run.json").write_text(json.dumps(state))
    log = world["workdir"] / "audit.log"
    os.utime(log, (finished_mtime, finished_mtime))


def test_a_detached_runtime_is_the_replay_not_the_wait(world):
    from datetime import UTC, datetime

    assert _run(world) == 0
    started = datetime(2026, 9, 12, 21, 24, 16, tzinfo=UTC)
    _as_detached(
        world, started=started.isoformat(), finished_mtime=started.timestamp() + 17 * 3600 + 39 * 60
    )

    assert _run(world, "--claim", "clm_detached_runtime") == 0

    receipt = json.loads((world["workdir"] / "result.json").read_text())
    assert receipt["runtime_seconds"] == 17 * 3600 + 39 * 60, (
        "the receipt billed the replay for the gap before it was reported"
    )
    assert receipt["finished_at"] is not None
    # And it is written back, so the next command does not have to re-derive it.
    assert json.loads((world["workdir"] / "run.json").read_text())["finished_at"]


def test_status_reports_the_real_finish_without_writing_to_the_workdir(world, capsys):
    """`status` is what an investigator runs on a workdir being kept as
    evidence. It must report honestly and change nothing."""
    from datetime import UTC, datetime

    assert _run(world) == 0
    started = datetime(2026, 9, 12, 21, 24, 16, tzinfo=UTC)
    _as_detached(world, started=started.isoformat(), finished_mtime=started.timestamp() + 3600)
    capsys.readouterr()

    before = (world["workdir"] / "run.json").read_text()
    assert cli.main(["status", "--workdir", str(world["workdir"]), "--no-color"]) == 0
    assert (world["workdir"] / "run.json").read_text() == before
