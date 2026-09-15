"""Is the replay alive, or merely unreaped?

`launch()` double-forks, so the replay is reparented to pid 1 — and in a
container pid 1 is routinely not an init that reaps (`sleep infinity`, a bash
wrapper). A finished replay then sits defunct for the life of the pod, and
signal 0 keeps succeeding on it, because the pid entry outlives the process
until its parent waits.

Calling that "running" is not a cosmetic error: `_completed_replay` refuses to
report a match the workdir already holds, `status` never reaches a verdict, and
`stop` signals a corpse. A 12-hour replay becomes permanently unreportable.

macOS cannot reproduce it — launchd reaps orphans — which is exactly why it
needs a test rather than a manual check.
"""

from __future__ import annotations

import pytest

from gensyn_audit import runner


def _stat(tmp_path, pid: int, line: bytes):
    directory = tmp_path / str(pid)
    directory.mkdir()
    (directory / "stat").write_bytes(line)
    return str(tmp_path)


def test_zombie_state_is_detected(tmp_path):
    proc = _stat(tmp_path, 5737, b"5737 (pretrain-audit-) Z 1 5737 0 0 -1 4194560\n")
    assert runner._is_zombie(5737, proc=proc) is True


@pytest.mark.parametrize("state", [b"R", b"S", b"D", b"T"])
def test_a_live_process_is_not_a_zombie(tmp_path, state):
    proc = _stat(tmp_path, 42, b"42 (pretrain-audit-) " + state + b" 1 42 0 0 -1 4194560\n")
    assert runner._is_zombie(42, proc=proc) is False


def test_comm_containing_parentheses_does_not_shift_the_state_field(tmp_path):
    """`comm` is arbitrary bytes between the FIRST '(' and the LAST ')'.

    Splitting on whitespace, or on the first ')', reads a field out of the
    process name and reports a zombie as running (or the reverse). The kernel's
    own advice is to scan from the end.
    """
    proc = _stat(tmp_path, 7, b"7 (weird ) name (x)) Z 1 7 0 0 -1 4194560\n")
    assert runner._is_zombie(7, proc=proc) is True

    proc = _stat(tmp_path, 8, b"8 (Z) (Z) R 1 8 0 0 -1 4194560\n")
    assert runner._is_zombie(8, proc=proc) is False


def test_absent_procfs_is_not_a_zombie(tmp_path):
    """No /proc at all (macOS), or a pid that vanished between the two reads.

    Guessing "zombie" here would report a live replay as finished and let a
    second one start on top of it, which is the more expensive mistake.
    """
    assert runner._is_zombie(5737, proc=str(tmp_path / "nonexistent")) is False


def test_is_running_reports_an_unreaped_process_as_gone(monkeypatch):
    monkeypatch.setattr(runner.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(runner, "_is_zombie", lambda pid: True)
    assert runner.is_running(5737) is False


def test_is_running_still_reports_a_live_process(monkeypatch):
    monkeypatch.setattr(runner.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(runner, "_is_zombie", lambda pid: False)
    assert runner.is_running(5737) is True


def test_is_running_rejects_a_nonsense_pid():
    assert runner.is_running(0) is False
    assert runner.is_running(-1) is False


# ── when did a detached replay actually finish? ──────────────────────────────
#
# 2026-09-13, from the first failed OPEN-1B audit's investigation: both audits'
# detached run.json carried a null `finished_at`, because `launch` returns as
# soon as the child starts and nothing reaps it. Every consumer then filled the
# gap with "now", so an overnight replay reported and submitted a runtime that
# included the hours the machine spent idle waiting for its auditor.

import json
import os
from datetime import UTC, datetime

from gensyn_audit import submit as submit_mod
from gensyn_audit.plan import Workdir

STARTED = "2026-09-12T21:24:16+00:00"


def _detached(tmp_path, *, pid: int, finished_at: str | None = None) -> tuple:
    wd = Workdir(tmp_path)
    wd.root.mkdir(parents=True, exist_ok=True)
    wd.log.write_text("2026-09-13 08:03:08,182 INFO pretrain.audit :: MATCH=False\n")
    # 17h39m after the recorded start, as the real one was.
    finish = datetime.fromisoformat(STARTED).timestamp() + 17 * 3600 + 39 * 60
    os.utime(wd.log, (finish, finish))
    state = runner.RunState(
        kit_id="k",
        kit_prefix="p",
        run="r",
        unit_kind="interval",
        config_name="c",
        until_step=103,
        audit_step=102,
        expect_hash="e" * 64,
        device="mps",
        pid=pid,
        argv=[],
        env_overlay={},
        started_at=STARTED,
        workdir=str(wd.root),
        detached=True,
        finished_at=finished_at,
    )
    return wd, state, finish


def test_a_detached_finish_is_recovered_from_the_log(tmp_path):
    wd, state, finish = _detached(tmp_path, pid=999_999)
    got = runner.observed_finish(state, wd)
    assert got is not None
    assert datetime.fromisoformat(got) == datetime.fromtimestamp(finish, UTC)


def test_a_running_replay_has_not_finished(tmp_path):
    wd, state, _ = _detached(tmp_path, pid=os.getpid())
    assert runner.observed_finish(state, wd) is None


def test_a_recorded_finish_is_never_overwritten(tmp_path):
    """`_run` supervises its own replay and already knows. Do not second-guess
    it from a log the reporting step may have appended to."""
    wd, state, _ = _detached(tmp_path, pid=999_999, finished_at="2026-09-13T15:03:08+00:00")
    assert runner.observed_finish(state, wd) == "2026-09-13T15:03:08+00:00"


def test_settle_persists_for_the_reporting_paths(tmp_path):
    wd, state, _ = _detached(tmp_path, pid=999_999)
    runner.save_state(wd, state)
    runner.settle_finish(state, wd)
    assert state.finished_at is not None
    assert json.loads(wd.state.read_text())["finished_at"] == state.finished_at


def test_settle_leaves_the_workdir_alone_when_asked(tmp_path):
    """`status` is the command an investigator runs on preserved evidence."""
    wd, state, _ = _detached(tmp_path, pid=999_999)
    runner.save_state(wd, state)
    before = wd.state.read_text()
    runner.settle_finish(state, wd, persist=False)
    assert state.finished_at is not None, "still reports it"
    assert wd.state.read_text() == before, "but must not write it"


def test_the_reported_runtime_is_the_replay_not_the_wait(tmp_path):
    wd, state, _ = _detached(tmp_path, pid=999_999)
    runner.settle_finish(state, wd)
    seconds = submit_mod._runtime_seconds(state)
    assert seconds == 17 * 3600 + 39 * 60


def test_an_unknown_runtime_is_reported_as_unknown_not_as_now(tmp_path):
    """No log to read: the honest answer is None, not the reporting time."""
    wd, state, _ = _detached(tmp_path, pid=999_999)
    wd.log.unlink()
    runner.settle_finish(state, wd)
    assert state.finished_at is None
    assert submit_mod._runtime_seconds(state) is None
