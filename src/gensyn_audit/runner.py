"""Launching the replay and living without it.

A step is twelve to twenty-four hours. That single fact decides the design:
the child must survive the terminal that started it, its state must live on
disk rather than in this process, and every other command must work by reading
that state rather than by talking to a running thing.

So `run --detach` starts a copy of itself in its own session with a log as its
only output, records what it started in `run.json`, and returns. That copy
replays, reports, submits and uploads. `status`, `logs` and `stop` are readers
of the state on disk; nothing outside the supervisor holds a handle on anything.
"""

from __future__ import annotations

import json
import os
import re
import resource
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .errors import AuditError, RunStateError
from .plan import OPEN_FILES, UNSET_VARS, Plan, Workdir


@dataclass
class RunState:
    """`run.json` — the handoff between the process that launched a replay and
    every process that asks about it later."""

    kit_id: str
    kit_prefix: str
    run: str
    unit_kind: str
    config_name: str
    until_step: int
    audit_step: int | None
    expect_hash: str
    device: str
    pid: int
    argv: list[str]
    env_overlay: dict[str, str]
    started_at: str
    workdir: str
    detached: bool = False
    supervisor_pid: int | None = None
    """The detached `gensyn-audit run` that owns this workdir end to end: it
    replays, then reports, submits and uploads without anyone re-running the
    command. `pid` is the replay it launched (0 until it has), and the two
    outlive each other in both directions, so both are recorded."""
    replay_python: str | None = None
    timed_out: bool = False
    """Version of the venv interpreter that produced the hash — NOT ours."""
    claim: str | None = None
    machine: str | None = None
    handle: str | None = None
    submission_id: str | None = None
    """The record's id for this result once a submit landed, with the claim it
    was submitted under. A claim token is spent by its submit, so a re-run that
    only needs to finish the hand-off must not POST the result again."""
    submitted_with: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return (
            f"{self.config_name} init"
            if self.unit_kind == "init"
            else f"{self.config_name} → step {self.until_step}"
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def _python_version(venv: Path) -> str | None:
    """The venv interpreter's version.

    Recorded because the receipt must name what produced the digest. Ours is
    whatever `gensyn-audit` was installed under and is not evidence about
    anything — reporting it would be a quiet lie in an audit artifact.
    """
    python = venv / "bin" / "python"
    if not python.is_file():
        return None
    proc = subprocess.run(
        [str(python), "-c", "import platform;print(platform.python_version())"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def observed_finish(state: RunState, workdir: Workdir) -> str | None:
    """When the replay actually stopped, for a run that recorded no finish.

    `_run` records `finished_at` because it supervises the replay itself. A
    detached launch returns as soon as the child is started and nothing ever
    reaps it, so its run.json keeps a null `finished_at` for good. Everything
    downstream then substituted "now": `submit._runtime_seconds` measured the
    replay against the moment the auditor came back to report it, which on an
    overnight audit added the idle hours to a 17-hour runtime, while `_report`
    showed 0s instead.

    The log's mtime is the last moment the replay wrote anything, and it is on
    the same epoch clock as `started_at`. The log's own timestamps are not:
    audit_replay formats asctime as naive local time, so subtracting one from a
    UTC `started_at` is the same class of error being fixed here.

    None while the replay is still going, or when the log cannot be read --
    unknown is a reportable answer and a wrong duration is not.
    """
    if state.finished_at:
        return state.finished_at
    if is_running(state.pid):
        return None
    try:
        mtime = workdir.log.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, UTC).isoformat(timespec="seconds")


def settle_finish(state: RunState, workdir: Workdir, *, persist: bool = True) -> RunState:
    """Fill in a missing `finished_at` from the log, in place.

    `persist=False` for read-only callers such as `status`, which must not
    write into a workdir someone is preserving as evidence.

    The exit code stays unknown either way: it is not recoverable after the
    fact, and the outcome is classified from the log's verdict anyway.
    """
    finish = observed_finish(state, workdir)
    if finish is None or finish == state.finished_at:
        return state
    state.finished_at = finish
    if persist:
        save_state(workdir, state)
    return state


def save_state(workdir: Workdir, state: RunState) -> None:
    workdir.root.mkdir(parents=True, exist_ok=True)
    workdir.state.write_text(state.to_json())


def load_state(workdir: Workdir) -> RunState:
    if not workdir.state.is_file():
        raise RunStateError(
            f"no audit has been started in {workdir.root}.",
            hint="Start one with `gensyn-audit run --step <N> --workdir <dir>`, or point "
            "--workdir at the directory you used.",
        )
    try:
        doc = json.loads(workdir.state.read_text())
    except json.JSONDecodeError as exc:
        raise RunStateError(f"{workdir.state} is corrupt: {exc}") from exc
    known = set(RunState.__dataclass_fields__)
    return RunState(**{k: v for k, v in doc.items() if k in known})


def _is_zombie(pid: int, *, proc: str = "/proc") -> bool:
    """Has `pid` exited without anyone reaping it?

    A detached replay runs in its own session, so it is reparented to pid 1 -- and in a
    container pid 1 is routinely not an init that reaps (`sleep infinity`, a
    bash wrapper), so a finished replay can sit defunct for the life of the
    pod. Signal 0 succeeds on a zombie: the pid entry outlives the process
    until its parent waits. Calling that "running" makes the workdir
    permanently unreportable -- `run` refuses to submit a match it already
    has, and `status` never reaches a verdict.

    Linux-only by design. It reads procfs, and on macOS launchd reaps orphans,
    so the state cannot persist there.
    """
    try:
        with open(f"{proc}/{pid}/stat", "rb") as fh:
            # `comm` may contain spaces and parentheses, so state is the first
            # field after the LAST ')'.
            return fh.read().rpartition(b")")[2].split()[0] == b"Z"
    except (OSError, IndexError):
        return False


def is_running(pid: int) -> bool:
    """Is the replay still alive?

    Signal 0 is the only portable probe, and it is honest about its limits: on
    a reboot the pid may belong to something else entirely, which is why
    `status` corroborates a live pid with a log that is still growing. It also
    cannot tell a running process from an unreaped one, which `_is_zombie`
    settles.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # alive, owned by someone else
    return not _is_zombie(pid)


def raise_open_files(target: int = OPEN_FILES) -> tuple[int, str | None]:
    """Raise RLIMIT_NOFILE for this process, so the child inherits it.

    The runbook says `ulimit -n 65536`; ~1700 shard memmaps blow through the
    macOS default of 256. This is the one place the tool changes anything about
    its environment, and it touches only the process about to be replaced.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= target:
        return soft, None
    ceiling = target if hard == resource.RLIM_INFINITY else min(target, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (ceiling, hard))
    except (ValueError, OSError) as exc:
        return soft, f"could not raise the open-file limit above {soft}: {exc}"
    if ceiling < target:
        return ceiling, (
            f"open-file limit raised to {ceiling}, short of the {target} the runbook asks "
            f"for (the hard limit is {hard}). Long fetches may still hit "
            '"too many open files"; `sudo launchctl limit maxfiles 65536 65536` lifts it.'
        )
    return ceiling, None


def build_env(plan: Plan) -> dict[str, str]:
    """The child's full environment: ours, plus the overlay, minus the strays."""
    env = dict(os.environ)
    env.update(plan.env_overlay())
    for key in UNSET_VARS:
        env.pop(key, None)
    return env


def _state_for(
    plan: Plan,
    pid: int,
    *,
    detached: bool,
    claim: str | None,
    machine: str | None,
    handle: str | None,
    supervisor_pid: int | None = None,
) -> RunState:
    return RunState(
        supervisor_pid=supervisor_pid,
        replay_python=_python_version(plan.venv),
        kit_id=plan.kit.kit_id,
        kit_prefix=plan.kit.public_prefix,
        run=plan.run,
        unit_kind=plan.unit.kind,
        config_name=plan.unit.config_name,
        until_step=plan.unit.until_step,
        audit_step=plan.audit_step,
        expect_hash=plan.unit.target_hash,
        device=plan.device,
        pid=pid,
        argv=plan.argv(),
        env_overlay=plan.env_overlay(),
        started_at=_now(),
        workdir=str(plan.workdir.root),
        detached=detached,
        claim=claim,
        machine=machine,
        handle=handle,
    )


def parse_duration(text: str) -> float:
    """`90`, `90s`, `30m`, `18h` -> seconds.

    A bare number is seconds. An audit's natural unit is hours, so `--timeout 18`
    meaning eighteen seconds would be a trap; it is spelled out rather than
    guessed at.
    """
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text or "")
    if not m:
        raise AuditError(
            f"could not read {text!r} as a duration.",
            hint="Use seconds, or a unit: 90s, 30m, 18h.",
        )
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2)]


def run_foreground(
    plan: Plan,
    *,
    claim: str | None = None,
    machine: str | None = None,
    handle: str | None = None,
    on_line=None,
    on_tick=None,
    timeout: float | None = None,
    supervisor_pid: int | None = None,
) -> tuple[RunState, bool]:
    """Run the replay attached, streaming its output to the log and to `on_line`.

    The PRD's shape is one command that prints its own progress and contacts the
    record when it finishes, so this is the default. An init unit takes under a
    minute; even an interval replay is fine attached on a machine that stays
    awake. `supervise` is for the case where it cannot: the same code, run by a
    detached copy of this tool, which passes its own pid as `supervisor_pid`.
    """
    plan.workdir.create()
    argv, env = plan.argv(), build_env(plan)
    if plan.needs_open_files:
        raise_open_files()
    timed_out = False

    with open(plan.workdir.log, "a", encoding="utf-8", errors="replace") as sink:
        sink.write(f"\n===== {plan.unit.label} started {_now()} =====\n$ {' '.join(argv)}\n\n")
        sink.flush()
        proc = subprocess.Popen(
            argv,
            cwd=str(plan.workdir.root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        state = _state_for(
            plan,
            proc.pid,
            detached=supervisor_pid is not None,
            claim=claim,
            machine=machine,
            handle=handle,
            supervisor_pid=supervisor_pid,
        )
        save_state(plan.workdir, state)

        # A watchdog rather than a poll: the read below blocks on the pipe, so
        # a deadline checked between lines would never fire on a replay that
        # has genuinely hung and stopped writing — which is the case a timeout
        # exists for.
        watchdog = None
        if timeout is not None:

            def _expire() -> None:
                nonlocal timed_out
                if proc.poll() is None:
                    timed_out = True
                    sink.write(
                        f"\n[gensyn-audit] deadline of {timeout:.0f}s reached; "
                        f"stopping the replay.\n"
                    )
                    sink.flush()
                    proc.kill()

            watchdog = threading.Timer(timeout, _expire)
            watchdog.daemon = True
            watchdog.start()

        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        try:
            while True:
                # A timed select rather than a blocking read: the status block
                # must keep ticking through the long silences between
                # micro-batches, and a replay that has stopped writing entirely
                # is exactly when the caller most wants to be told.
                ready, _, _ = select.select([fd], [], [], 0.4)
                if ready:
                    chunk = os.read(fd, 65536)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    sink.write(text)
                    sink.flush()
                    if on_line:
                        for line in text.splitlines():
                            if line.strip():
                                on_line(line)
                if on_tick:
                    on_tick()
            proc.wait()
        finally:
            if watchdog is not None:
                watchdog.cancel()

    state.exit_code = proc.returncode
    state.finished_at = _now()
    state.timed_out = timed_out
    save_state(plan.workdir, state)
    return state, timed_out


#: The key under which `supervise` hands the claim token to its child. It
#: travels down a pipe rather than the command line: argv is visible to every
#: user on the machine for as long as the supervisor runs, and is echoed into
#: `gensyn-audit.log`.
HANDOFF_CLAIM = "claim"


def redact_argv(argv: list[str]) -> list[str]:
    """`argv` with the value of any `--claim` blanked, for logs and screens."""
    out: list[str] = []
    hide = False
    for tok in argv:
        if hide:
            out.append("<redacted>")
            hide = False
        elif tok == "--claim":
            out.append(tok)
            hide = True
        elif tok.startswith("--claim="):
            out.append("--claim=<redacted>")
        else:
            out.append(tok)
    return out


def read_handoff(stream=None) -> dict:
    """What `supervise` sent this process, if anything.

    The child blocks here until its parent closes the pipe, which the parent
    does only after `run.json` names the child as supervisor. That ordering is
    the point: it is what stops the parent's write from landing on top of the
    child's own, which would erase the replay pid the child had just recorded.
    A terminal, a closed descriptor or a captured stdin means no parent: the
    invocation is being driven by hand, and there is nothing to wait for.
    """
    stream = sys.stdin if stream is None else stream
    try:
        if stream is None or getattr(stream, "closed", False) or stream.isatty():
            return {}
        text = stream.read()
    except (OSError, ValueError, AttributeError):
        return {}
    if not text or not text.strip():
        return {}
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return doc if isinstance(doc, dict) else {}


def supervise(
    plan: Plan,
    argv: list[str],
    *,
    claim: str | None = None,
    machine: str | None = None,
    handle: str | None = None,
) -> RunState:
    """Hand the rest of this audit to a detached copy of the tool.

    The first detached design backgrounded only the replay and asked the
    auditor to come back and re-run the same command to report and submit it.
    Every tester tripped on that step: it is the one part of the flow that
    cannot be inferred from watching the screen. So what detaches now is
    `gensyn-audit run` itself -- `argv` is this invocation minus `--detach`,
    plus the flag that tells the child it is the supervisor -- and the replay,
    the verdict, the submission and the upload all happen in that process.
    Nothing needs re-running unless it dies, in which case the old recovery
    path (run the same command again) still works and is what `status` says.

    It gets its own session, so closing the terminal or the SSH connection does
    not take it down, and it keeps this process's working directory, so every
    relative path on the command line still means what it meant here. Its
    screen goes to `gensyn-audit.log` in the workdir; the replay's own output
    still goes to `audit.log`.

    Two things go to the child through its stdin rather than its argv, and
    the order matters. `run.json` is written naming the child as supervisor
    *before* the pipe is closed, and the child does not move until it is: so
    the parent's one write can never land on top of the child's, and `stop`
    can never find a replay pid that has been overwritten with 0. And the
    claim token rides in the same message, because argv is public on a shared
    machine for the whole replay and is echoed into the log.
    """
    workdir = plan.workdir
    workdir.create()
    env = dict(os.environ)
    # The child's stdout is a file. Unbuffered so `status` sees the current
    # phase rather than the last 8 KB boundary, and UTF-8 so the marks it
    # prints are the ones a terminal would have shown.
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    command = [sys.executable, "-m", "gensyn_audit.cli", *argv]
    shown = " ".join(redact_argv(command))
    with open(workdir.cli_log, "ab") as sink:
        sink.write(f"\n===== supervisor started {_now()} =====\n$ {shown}\n\n".encode())
        sink.flush()
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=sink,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            start_new_session=True,
        )
    # pid 0: no replay yet. The child overwrites this the moment it launches
    # one, carrying the supervisor pid forward; until then `status` reports the
    # supervisor as preparing. The child is still waiting on its stdin while
    # this is written, which is what makes the overwrite one-directional.
    state = _state_for(
        plan, 0, detached=True, claim=claim, machine=machine, handle=handle, supervisor_pid=proc.pid
    )
    save_state(workdir, state)
    handoff = {HANDOFF_CLAIM: claim} if claim else {}
    assert proc.stdin is not None
    try:
        proc.stdin.write(json.dumps(handoff).encode())
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        # The child is already gone. `status` will say it ended before it
        # could report; nothing to do about it from here.
        pass
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
    return state


def supervisor_running(state: RunState) -> bool:
    """Is the detached `run` that owns this workdir still alive? Never true of
    the current process: the supervisor itself asks this too."""
    pid = state.supervisor_pid
    return bool(pid) and pid != os.getpid() and is_running(pid)


def stop(state: RunState, workdir: Workdir, *, force: bool = False) -> bool:
    """Ask the replay to stop. Returns False if it was not running.

    The supervisor goes first. Left alive, it would see its replay die,
    classify the result as canceled and -- with a claim -- submit that for
    triage, which is not what someone who typed `stop` meant.
    """
    stopped = False
    supervisor = state.supervisor_pid
    if supervisor is not None and supervisor_running(state):
        os.kill(supervisor, signal.SIGKILL if force else signal.SIGTERM)
        for _ in range(50):
            if not is_running(supervisor):
                break
            time.sleep(0.1)
        stopped = True
    if is_running(state.pid):
        os.kill(state.pid, signal.SIGKILL if force else signal.SIGTERM)
        if not force:
            for _ in range(50):
                if not is_running(state.pid):
                    break
                time.sleep(0.2)
        stopped = True
    if not stopped:
        return False
    state.finished_at = _now()
    state.exit_code = -1
    save_state(workdir, state)
    return True


def follow(path: Path, *, from_start: bool = False, poll: float = 0.5):
    """Yield log text as it is written, `tail -f` style.

    Deliberately a generator over raw chunks rather than lines: tqdm's
    carriage-return redraws never terminate a line, and buffering for one would
    freeze the display for hours.
    """
    while not path.is_file():
        time.sleep(poll)
    with open(path, "r", errors="replace") as fh:
        if not from_start:
            fh.seek(0, os.SEEK_END)
        while True:
            chunk = fh.read()
            if chunk:
                yield chunk
            else:
                time.sleep(poll)
