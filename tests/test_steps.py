"""Two step numberings, differing by one.

The record and the web app count auditable steps from zero; the run counts
trained steps from one, with step 0 as init. Every artifact on disk —
checkpoint directories, `state_hashes.jsonl`, `metrics.jsonl`, `meta.json`,
and `audit_replay --until-step` — uses the run's numbering. Only the record
uses the other.

Conflating them costs an auditor a full day and reads as a reproducibility
failure rather than an indexing bug, which is why the conversion is a type.
"""

from __future__ import annotations

import pytest

from gensyn_audit.steps import StepRef


def test_audit_step_is_one_below_the_log_step():
    assert StepRef.from_audit(25700).log == 25701
    assert StepRef.from_log(25701).audit == 25700


@pytest.mark.parametrize(
    "audit,segment,position,closes",
    [
        (0, 0, 1, False),
        (99, 0, 100, True),  # closes segment 0 — log step 100, a published checkpoint
        (100, 1, 1, False),
        (25699, 256, 100, True),
        (25700, 257, 1, False),
        (25799, 257, 100, True),
    ],
)
def test_segment_arithmetic_matches_the_record(audit, segment, position, closes):
    """Checked against live receipts: the record returns exactly these."""
    s = StepRef.from_audit(audit)
    assert (s.segment, s.position, s.closes_segment) == (segment, position, closes)


def test_a_closing_step_lands_on_a_published_checkpoint():
    """Position 100 of a segment produces the state the run published — which
    is the whole reason the record numbers from zero."""
    s = StepRef.from_audit(99)
    assert s.closes_segment
    assert s.log == 100, "log step 100 is published checkpoint 100"


def test_the_predecessor_is_the_state_before():
    """Numerically equal to the audit number, which is a coincidence of the
    offset being one — not a licence to use them interchangeably."""
    s = StepRef.from_audit(25700)
    assert s.predecessor_log == 25700
    assert s.log == 25701
    assert f"step_{s.predecessor_log:09d}" == "step_000025700"


def test_init_is_not_auditable():
    """Log step 0 is init: regenerated from a seed, not replayed from a
    predecessor. There is no audit step below zero."""
    with pytest.raises(ValueError, match="not auditable"):
        StepRef.from_log(0)


def test_the_two_numbers_read_differently():
    assert str(StepRef.from_audit(25700)) == "audit step 25700 (log step 25701)"
