"""Two step numberings, and the conversion between them.

The run and the record count steps differently, and the difference is exactly
one. Getting it wrong costs an auditor a day and looks like a reproducibility
failure rather than an indexing bug, so the conversion lives in one place and
the two numbers are never plain ints in the same scope.

**Log numbering** is what the run produced. ``state_hashes.jsonl`` has step 0
as init and 1..80,957 as trained steps. The same numbers name the checkpoint
directories (``step_000025700``), the ``step`` field in each ``meta.json``, the
rows of ``metrics.jsonl``, and ``audit_replay --until-step``.

**Audit numbering** is what the record and the web app use: the auditable steps
counted from zero, so audit step N is log step N+1. That makes
``segment = N // 100`` and a segment close at position 100 land on a published
checkpoint, which is why it was chosen.

A volunteer copies an audit number off the web app. Everything the runner then
touches on disk speaks log numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

#: log step = audit step + this.
OFFSET = 1


@dataclass(frozen=True)
class StepRef:
    """One step, in both numberings. Construct from whichever you were given."""

    audit: int
    """As the record and the web app number it. What a claim is issued for."""

    @classmethod
    def from_audit(cls, n: int) -> StepRef:
        return cls(audit=n)

    @classmethod
    def from_log(cls, n: int) -> StepRef:
        if n < OFFSET:
            raise ValueError(f"log step {n} is init or earlier; it is not auditable")
        return cls(audit=n - OFFSET)

    @property
    def log(self) -> int:
        """The state this audit produces: ``--until-step``, and the row in
        ``state_hashes.jsonl`` / ``metrics.jsonl`` that describes it."""
        return self.audit + OFFSET

    @property
    def predecessor_log(self) -> int:
        """The state it starts from — the checkpoint directory to load.

        Numerically the same as the audit number, which is a coincidence of the
        offset being one and not a reason to conflate them.
        """
        return self.audit

    @property
    def is_genesis(self) -> bool:
        """The one audit whose predecessor is not a checkpoint.

        Audit 0 replays log step 1, and log step 0 is the run's initialization:
        regenerated from the seed, hashed, and published as
        ``ckpt/state_hash_init.txt`` -- never written as a checkpoint
        directory. ``ckpt/step_000000000/`` does not exist and never did; the
        run's checkpoints start at 100.

        So this step alone starts from ``--from-init`` rather than a download.
        Everything after it is an ordinary interval.
        """
        return self.audit == 0

    @property
    def segment(self) -> int:
        return self.audit // 100

    @property
    def position(self) -> int:
        """1-based position within the segment; 100 closes it."""
        return self.audit % 100 + 1

    @property
    def closes_segment(self) -> bool:
        return self.position == 100

    def __str__(self) -> str:
        return f"audit step {self.audit} (log step {self.log})"
