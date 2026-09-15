"""Build the local result and receipt for a completed replay.

The receipt records the outcome, machine, runtime, and produced digest whether
or not the replay matched. ``record.py`` submits the same result to the record
service when the user provides a claim token.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from . import __version__
from .plan import Workdir
from .progress import Progress, read_loss_log
from .runner import RunState


@dataclass
class Divergence:
    """Where two digests part company, for the `character N` reading."""

    first: int

    @property
    def display(self) -> str:
        return f"diverges from character {self.first + 1}"


def divergence(got: str, want: str) -> Divergence | None:
    if got == want:
        return None
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            return Divergence(i)
    return Divergence(min(len(got), len(want)))


@dataclass
class Result:
    """The receipt. Everything here is measured or declared; nothing inferred.

    The provenance block is not decoration: a state hash is a claim about
    repop's kernels as much as about the model code, so a receipt that cannot
    name the kernel build that produced it is not audit evidence.
    """

    schema: int
    tool: str
    kit_id: str
    kit_prefix: str
    unit_kind: str
    run: str
    outcome: str
    """One of the five. Only `match` can advance the record."""

    config_name: str
    until_step: int
    audit_step: int | None
    matched: bool | None
    reproduced_hash: str | None
    committed_hash: str
    repop_commit: str | None
    repop_backends: list[str]
    device: str | None
    losses: list[dict]
    started_at: str
    finished_at: str | None
    runtime_seconds: float | None
    machine: str | None
    handle: str | None
    claim: str | None
    host: dict
    artifact: dict
    diverges_at: int | None
    exit_code: int | None
    log_pointer: str | None = None
    mocked: dict | None = None
    """Present only when something in this run was stood in for. A reader
    must never have to guess whether a receipt describes a real service."""

    predecessor: dict | None = None
    """What was established about the STARTING checkpoint, in the three
    separate terms of `verify.Verdict`. A replay's own match says nothing
    about where it started from, so a reader who cannot see this cannot tell
    an audit from an audit of an unverified anchor."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def _host_facts(device: str, replay_python: str | None) -> dict:
    facts: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        # The interpreter the REPLAY ran under, from the kit's venv. Not ours:
        # gensyn-audit's own interpreter had no part in producing the digest.
        "python": replay_python,
        "device": device,
    }
    try:
        import os

        facts["memory_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
        )
    except (ValueError, OSError, AttributeError):
        pass
    return facts


def _runtime_seconds(state: RunState) -> float | None:
    """The replay's duration, or None. Never measured against the present.

    A detached launch records no finish of its own, and falling back to "now"
    here billed the receipt for every hour between the replay ending and the
    auditor coming back to report it. `runner.settle_finish` recovers the real
    finish from the log before this runs; if it could not, None is the honest
    answer.
    """
    if not state.finished_at:
        return None
    try:
        start = datetime.fromisoformat(state.started_at)
        end = datetime.fromisoformat(state.finished_at)
        return round((end - start).total_seconds(), 1)
    except (TypeError, ValueError):
        return None


def build(
    state: RunState,
    progress: Progress,
    workdir: Workdir,
    *,
    artifact_url: str | None = None,
    outcome=None,
    record_provenance: dict | None = None,
    bundle: Path | None = None,
) -> Result:
    reproduced = progress.state_hash
    diff = divergence(reproduced, state.expect_hash) if reproduced else None

    # An init unit regenerates state from a seed and hands nothing on; an
    # empty artifact block would imply otherwise.
    handoff = Path(progress.saved_checkpoint) if progress.saved_checkpoint else workdir.handoff
    artifact: dict = {} if state.unit_kind == "init" else {"kind": "chained-audit checkpoint"}
    if state.unit_kind != "init" and bundle is not None and bundle.is_file():
        # The packed hand-off, not the directory it came from. Its digest is
        # what the record is told at submit as `artifactDigest` and what the
        # verifier checks the arriving bytes against, so the two must describe
        # the same file.
        from .upload import digest_file

        artifact["path"] = str(bundle)
        artifact["bytes"] = bundle.stat().st_size
        artifact["digest"] = digest_file(bundle)
        artifact["source"] = str(handoff)
    elif state.unit_kind != "init" and handoff.exists():
        artifact["path"] = str(handoff)
        artifact["bytes"] = sum(f.stat().st_size for f in handoff.rglob("*") if f.is_file())
    if artifact_url:
        artifact["url"] = artifact_url

    # Prefer the loss log: it is written per step and survives a killed
    # process, where the result object only exists if the process exited.
    losses = progress.rank0_losses or read_loss_log(workdir.loss_log)

    return Result(
        schema=1,
        tool=f"gensyn-audit/{__version__}",
        kit_id=state.kit_id,
        kit_prefix=state.kit_prefix,
        run=state.run,
        outcome=(
            outcome.value if outcome is not None else ("match" if progress.match else "no-match")
        ),
        unit_kind=state.unit_kind,
        config_name=state.config_name,
        until_step=state.until_step,
        audit_step=state.audit_step,
        matched=progress.match,
        reproduced_hash=reproduced,
        committed_hash=state.expect_hash,
        repop_commit=progress.repop_commit,
        repop_backends=list(progress.repop_backends),
        device=progress.device or state.device,
        losses=losses,
        started_at=state.started_at,
        finished_at=state.finished_at,
        runtime_seconds=_runtime_seconds(state),
        machine=state.machine,
        handle=state.handle,
        claim=state.claim,
        host=_host_facts(state.device, state.replay_python),
        artifact=artifact,
        diverges_at=diff.first if diff else None,
        exit_code=state.exit_code,
        log_pointer=str(workdir.log),
        mocked=record_provenance,
    )


def write(result: Result, workdir: Workdir) -> Path:
    workdir.result.write_text(result.to_json())
    return workdir.result
