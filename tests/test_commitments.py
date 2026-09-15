"""Committed hashes come from the run's published log, not from an API row.

The record derives placeholders until its pipeline publishes, and a replay
compared against one cannot match however perfect it was — so where a published
state-hash log exists, it wins.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from gensyn_audit import commitments
from gensyn_audit.errors import AuditError

H_25700 = "556bb81697a3952933b54a7b41e47add3da42ea8e66e10569cb172b4aea2e650"
H_25701 = "50719216326e684606c239efa146a6af093af4756a49ca37dc9f83e5ee3d2519"


def _log(tmp_path, rows) -> str:
    p = tmp_path / "state_hashes.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_reads_a_state_hash_log(tmp_path):
    src = _log(
        tmp_path,
        [
            {"step": 25700, "consumed_tokens": 1, "state_hash": H_25700},
            {"step": 25701, "consumed_tokens": 2, "state_hash": H_25701},
        ],
    )
    c = commitments.load(src)
    assert c.get(25701) == H_25701
    assert c.require(25700) == H_25700


def test_a_resumed_run_keeps_the_last_occurrence(tmp_path):
    """Resumed runs re-log overlapping steps; the surviving segment is the one
    that ran, matching how the stitched logs read."""
    src = _log(
        tmp_path,
        [
            {"step": 25701, "state_hash": "a" * 64},
            {"step": 25701, "state_hash": H_25701},
        ],
    )
    assert commitments.load(src).get(25701) == H_25701


def test_an_unhashed_step_says_why(tmp_path):
    """A digest exists at step k only when every_n_steps divides k."""
    src = _log(
        tmp_path, [{"step": 100, "state_hash": "a" * 64}, {"step": 200, "state_hash": "b" * 64}]
    )
    with pytest.raises(AuditError) as exc:
        commitments.load(src).require(150)
    assert "100–200" in exc.value.hint
    assert "every_n_steps" in exc.value.hint


def test_a_truncated_tail_is_skipped_not_fatal(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps({"step": 1, "state_hash": "c" * 64}) + '\n{"step": 2, "sta')
    assert commitments.load(str(p)).get(1) == "c" * 64


def test_rows_without_a_full_digest_are_ignored(tmp_path):
    """The init row carries `kind: init`; a short digest is not a commitment."""
    src = _log(
        tmp_path,
        [
            {"step": 0, "kind": "init", "state_hash": "d" * 64},
            {"step": 1, "state_hash": "tooshort"},
        ],
    )
    c = commitments.load(src)
    assert c.get(0) == "d" * 64
    assert c.get(1) is None


def test_an_empty_log_is_refused(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n\n")
    with pytest.raises(AuditError, match="no state hashes"):
        commitments.load(str(p))


# ── which source wins ────────────────────────────────────────────────────────


def test_the_published_log_wins_over_the_api():
    seen = []
    got = commitments.reconcile(
        25701, published=H_25701, from_api="9" * 64, on_conflict=lambda p, a: seen.append((p, a))
    )
    assert got == H_25701
    assert seen == [(H_25701, "9" * 64)], "a disagreement must be reported, not hidden"


def test_agreement_is_silent():
    seen = []
    got = commitments.reconcile(
        25701, published=H_25701, from_api=H_25701, on_conflict=lambda p, a: seen.append(1)
    )
    assert got == H_25701 and not seen


def test_the_api_is_used_when_no_log_is_named():
    assert commitments.reconcile(25701, published=None, from_api=H_25701) == H_25701


def test_neither_source_is_an_error():
    with pytest.raises(AuditError) as exc:
        commitments.reconcile(25701, published=None, from_api=None)
    assert "artifacts.state_hashes" in exc.value.hint


# ── inclusion proofs ─────────────────────────────────────────────────────────

# Served by the record on 2026-09-09 for step 80956, kept verbatim: a tree we
# only check against itself would agree with a wrong construction. Its leaf is
# promoted through the first three levels, so the path starts at level 3 and is
# shorter than a full segment's.
SERVED = commitments.Proof(
    leaf="b656715341005431b657042128671c23d3c2cf6c5503b91bb62134ca5c3b023d",
    root="2a4b64be6aa55a2dd0d5df3e573a4248b7e1cf3eaf496aaa85efdad1a2f52caf",
    path=(
        commitments.ProofLink(
            "768dd4e2d54529c0f4df9a13ad78a54808c51fcd5a85bacff17c9c6734d7df11", "left"
        ),
        commitments.ProofLink(
            "8beb3440069838d0a5ae23c2808a74f6a9bd07d8fcbf1183e985bd8c954ee5d8", "left"
        ),
        commitments.ProofLink(
            "89bc08e939f1a4473de8d5c312259fad8d312b681eda96407c3cc7688ee56a20", "left"
        ),
    ),
)


def test_verifies_a_proof_the_record_served():
    assert commitments.verify_inclusion(SERVED)


def test_rejects_a_forged_sibling():
    forged = replace(SERVED, path=(replace(SERVED.path[0], hash="11" * 32),) + SERVED.path[1:])
    assert not commitments.verify_inclusion(forged)


def test_rejects_a_flipped_side():
    # Order matters inside the pair, or a prover could reorder a subtree.
    flipped = replace(SERVED, path=(replace(SERVED.path[0], side="right"),) + SERVED.path[1:])
    assert not commitments.verify_inclusion(flipped)


def test_rejects_a_leaf_that_is_not_in_the_tree():
    assert not commitments.verify_inclusion(replace(SERVED, leaf="22" * 32))


def test_a_malformed_digest_fails_rather_than_raising():
    assert not commitments.verify_inclusion(replace(SERVED, leaf="not-a-digest"))
