"""What a complete hand-off is, in one place.

A hand-off is not "a checkpoint". It is a checkpoint *plus* the three things
the next auditor needs and a resume does not: the target step's gradients
(`gradients.safetensors`), every DP rank's batch-hasher chain, and
the descriptor that says how many of those there should be. Without them the
recipient can load the state and replay from it, but cannot reconstruct the
hash the run published for it — which is the difference between continuing a
chain and continuing someone's word for it.

So "it downloaded" and "it is usable" have to be the same question, asked in
one place: the uploader must not send a partial hand-off, the downloader must
not treat one as verified, and the preflight must size disk for all of it.

The member list mirrors what ``audit_replay --save-checkpoint-dir`` writes
(``_save_chained_audit_checkpoint``) and what it reads back on the next
interval. `spike_protocol.json` and `sampler.rank_*.json` are deliberately not
required: the writer omits them for an auditable run with no spike state, and
the reader tolerates their absence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import AuditError

#: The final gradients exported with the checkpoint.
GRADIENTS = "gradients.safetensors"

#: Written last by the checkpoint writer. Its absence means the writer never
#: finished or the transfer did not — never "an older format".
COMPLETE = "_COMPLETE"

#: Present in every hand-off regardless of topology.
_ALWAYS = (
    "meta.json",
    COMPLETE,
    "state_hash.txt",
    "global_stream.json",
    "rng.rank_0.pt",
    GRADIENTS,
)

#: Upper bound on a declared dp_world_size. `meta.json` is the artifact
#: author's, so it decides how many files we go looking for; mirrors
#: audit_replay's own `_MAX_AUDIT_WORLD`.
MAX_WORLD = 48


@dataclass(frozen=True)
class Handoff:
    """One hand-off directory, and whether it is all there."""

    root: Path
    step: int | None
    dp_world_size: int
    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing

    @property
    def files(self) -> tuple[Path, ...]:
        """Every file in the hand-off, sorted — what must actually travel."""
        return tuple(sorted(p for p in self.root.rglob("*") if p.is_file()))

    @property
    def has_gradients(self) -> bool:
        return GRADIENTS in self.present


def _descriptor(root: Path) -> tuple[int | None, int]:
    """`(step, dp_world_size)` from meta.json.

    An unreadable meta is an error, not a default: falling back to one rank
    would let a hand-off from a 32-rank run pass the completeness check while
    carrying only rank 0's batch chain — the file that decides what to look for
    cannot be the one we shrug at.
    """
    import json

    if not (root / "meta.json").is_file():
        return None, 1  # reported as a missing member by the caller
    try:
        meta = json.loads((root / "meta.json").read_text())
    except (OSError, ValueError) as exc:
        raise AuditError(
            f"{root}/meta.json is not readable JSON: {exc}",
            hint="It names the topology, which decides how many per-rank "
            "batch-hasher chains a complete hand-off has. Without it a "
            "32-rank hand-off carrying one chain looks complete.",
        ) from exc
    if not isinstance(meta, dict):
        raise AuditError(f"{root}/meta.json must contain an object.")
    world = meta.get("dp_world_size")
    if type(world) is not int or not 1 <= world <= MAX_WORLD:
        raise AuditError(
            f"{root}/meta.json declares dp_world_size={world}; the cap is {MAX_WORLD}.",
            hint="meta.json comes from whoever produced the artifact, and this "
            "value decides how many per-rank files are read. A value out "
            "of range is corruption or an attempt at one.",
        )
    step = meta.get("step")
    if type(step) is not int or step <= 0:
        raise AuditError(f"{root}/meta.json must declare a positive integer step.")
    return step, world


def inspect(root: Path) -> Handoff:
    """Enumerate a hand-off directory against the member contract."""
    if not root.is_dir():
        raise AuditError(f"no hand-off directory at {root}.")
    step, world = _descriptor(root)
    wanted = [*_ALWAYS, *(f"batch_hasher.rank_{r}.bin" for r in range(world))]
    present = tuple(n for n in wanted if (root / n).is_file())
    missing = tuple(n for n in wanted if n not in present)
    if not (root / "dcp").is_dir() and not (root / "fallback.pt").is_file():
        missing = (*missing, "dcp/")
    return Handoff(root=root, step=step, dp_world_size=world, present=present, missing=missing)


def require_complete(h: Handoff, *, what: str) -> None:
    """Refuse an incomplete hand-off, naming what is absent and what it costs."""
    if h.is_complete:
        return
    detail = {
        GRADIENTS: "the target step's final gradients — the published state "
        "hash folds them in, so the recipient cannot reconstruct it",
        COMPLETE: "the writer's completion marker — this checkpoint was never "
        "published, or the transfer stopped short",
        "dcp/": "the model and optimizer shards themselves",
    }
    lines = []
    for name in h.missing:
        why = detail.get(name) or (
            "a per-rank batch-hasher chain the run's batch digest is built from"
            if name.startswith("batch_hasher.")
            else "required hand-off metadata"
        )
        lines.append(f"  {name} — {why}")
    raise AuditError(
        f"{what} is incomplete: {len(h.missing)} required file(s) missing from {h.root}.",
        hint="\n".join(lines) + "\n"
        "A hand-off that cannot be checked against the published state "
        "hash must not be replayed from or passed on as if it had been.",
    )
