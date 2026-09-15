"""Submission, the loss gate, and upload — against the mock record.

Nothing answers the PRD's HTTP contract yet, so these drive `MockRecord`, which
implements the same protocol. The gate it applies is the real one: exact
equality against a withheld loss series, which is what makes these tests worth
anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gensyn_audit import upload as uploadmod
from gensyn_audit.errors import AuditError
from gensyn_audit.mock import MockRecord
from gensyn_audit.outcome import Outcome, classify
from gensyn_audit.record import result_bundle

RUN = "20260703-171943-4e85cd3"
COMMITTED = "49a35d246590678b78088163da5c4600591e203d72beae008419ec5e479bdc2a"
PRED_DIGEST = "16b3d656df3b2c9ac9dd3dacdfad3a0132f7e17c0c175c0f369eae519d776e30"
# Synthetic. The real withheld series is never committed to this repo: the
# loss gate's whole strength is that those values were not published.
TRUE_CE = 4.8151623042000000
TRUE_ZL = 0.0234567890123456


@pytest.fixture
def record(tmp_path: Path) -> MockRecord:
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "run": RUN,
                "gcs_root": "gs://bucket/data/shards",
                "steps": {
                    "200": {
                        "committed_hash": COMMITTED,
                        "predecessor": {
                            "source": "published",
                            "step": 100,
                            "uri": "gs://bucket/checkpoints/step_000000100",
                            "digest": PRED_DIGEST,
                        },
                        "phase": "warmup",
                        "microbatches": 192,
                    }
                },
                "withheld_losses": {"200": {"loss_ce": TRUE_CE, "loss_zloss": TRUE_ZL}},
            }
        )
    )
    return MockRecord(fixture)


def _bundle(outcome: Outcome, *, ce=TRUE_CE, zl=TRUE_ZL, hash_=COMMITTED, artifact=None) -> dict:
    return result_bundle(
        outcome=outcome,
        receipt={
            "tool": "gensyn-audit/0.1.0",
            "kit_id": "pt-aaa_rp-bbb",
            "kit_prefix": "gs://k",
            "repop_commit": "c" * 40,
            "config_name": RUN,
            "until_step": 200,
            "reproduced_hash": hash_,
            "committed_hash": COMMITTED,
            "losses": [{"step": 200, "loss_ce": ce, "loss_zloss": zl}],
            "device": "mps",
            "machine": "M4 Pro · 24 GB",
            "handle": "someone",
            "host": {},
            "runtime_seconds": 43200.0,
            "log_pointer": "/w/audit.log",
            "artifact": artifact or {},
        },
    )


def test_the_submit_body_uses_the_apis_field_names():
    """These are the wire names in the contract; a typo here is a 422 after a
    twelve-hour replay."""
    b = _bundle(Outcome.MATCH)
    assert b["outcome"] == "match"
    assert b["reportedStateHash"] == COMMITTED
    assert b["ce"] == TRUE_CE and b["zLoss"] == TRUE_ZL
    assert b["runtimeHours"] == 12.0
    assert b["hardware"] == "M4 Pro · 24 GB"


# ── step context ─────────────────────────────────────────────────────────────


def test_step_context_carries_the_predecessor_and_its_digest(record):
    ctx = record.step_context(RUN, 200)
    assert ctx.committed_hash == COMMITTED
    assert ctx.predecessor.step == 100, "checkpoints are periodic, not per-step"
    assert ctx.predecessor.digest == PRED_DIGEST
    assert ctx.predecessor.source == "published"
    assert not ctx.predecessor.is_crowd_provided


def test_an_unknown_step_names_what_the_record_has(record):
    with pytest.raises(AuditError, match="no context for step 999"):
        record.step_context(RUN, 999)


# ── the loss gate ────────────────────────────────────────────────────────────


def test_a_real_replay_is_recorded_pending_verification(record):
    """Nothing is accepted synchronously: the loss gate runs elsewhere, so the
    honest report is `recorded`, not `accepted`."""
    r = record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_a")
    assert r.disposition == "pending-verification"
    assert r.recorded and r.receipt_url
    assert r.public_state_changed is False


def test_a_hash_without_losses_is_refused(record):
    """The committed hash is public. Echoing it proves nothing."""
    b = _bundle(Outcome.MATCH)
    b.pop("ce", None)
    b.pop("zLoss", None)
    r = record.submit(RUN, b, claim="clm_b")
    assert not r.recorded
    assert "not evidence of work" in r.detail


def test_a_loss_that_is_merely_close_is_refused(record):
    """Under bitwise reproducibility they are identical. A tolerance here would
    accept a replay that was not one."""
    r = record.submit(RUN, _bundle(Outcome.MATCH, ce=TRUE_CE + 1e-12), claim="clm_c")
    assert not r.recorded
    assert "ce does not match" in r.detail


def test_zloss_is_gated_too_not_just_ce(record):
    r = record.submit(RUN, _bundle(Outcome.MATCH, zl=TRUE_ZL * 1.0001), claim="clm_d")
    assert not r.recorded and "zLoss" in r.detail


# ── outcomes: only a match advances ──────────────────────────────────────────


@pytest.mark.parametrize(
    "outcome",
    [
        Outcome.NO_MATCH,
        Outcome.TIMEOUT,
        Outcome.CANCELED,
        Outcome.INCONCLUSIVE,
    ],
)
def test_no_other_outcome_changes_public_state(record, outcome):
    r = record.submit(RUN, _bundle(outcome), claim=f"clm_{outcome.value}")
    assert r.disposition == "recorded-no-public-change"
    assert not r.public_state_changed
    assert "200" not in record.state.accepted
    assert not outcome.advances_record


def test_a_failed_submission_does_not_spend_the_claim(record):
    """The step returns to the pool; the auditor can try again."""
    record.submit(RUN, _bundle(Outcome.NO_MATCH), claim="clm_e")
    assert "clm_e" not in record.state.used_claims
    assert record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_e").recorded


def test_classify_prefers_the_reason_over_the_symptom():
    """Cancel and timeout both produce no digest, but reporting either as
    inconclusive would lose the one fact that explains it."""
    assert classify(matched=None, exit_code=1, interrupted=True) is Outcome.CANCELED
    assert classify(matched=None, exit_code=1, timed_out=True) is Outcome.TIMEOUT
    assert classify(matched=None, exit_code=0) is Outcome.INCONCLUSIVE
    assert classify(matched=True, exit_code=0) is Outcome.MATCH
    assert classify(matched=False, exit_code=1) is Outcome.NO_MATCH
    # A cancel wins even when an earlier attempt left a verdict in the log.
    assert classify(matched=True, exit_code=0, interrupted=True) is Outcome.CANCELED


def test_exit_codes_separate_disagreed_from_never_ran():
    assert Outcome.MATCH.exit_code == 0
    assert Outcome.NO_MATCH.exit_code == Outcome.TIMEOUT.exit_code == 2
    assert Outcome.CANCELED.exit_code == Outcome.INCONCLUSIVE.exit_code == 1


# ── single-use, idempotency, superseded ──────────────────────────────────────


def test_a_spent_claim_is_refused_and_points_at_the_receipt(record):
    """The API is deliberately NOT idempotent: the token is single-use, so a
    retry after a lost response gets claim_not_active and the auditor is told
    how to finish the hand-off instead. A forgiving mock would hide the case
    the CLI most needs to handle well."""
    record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_f")
    with pytest.raises(AuditError) as exc:
        record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_f")
    assert "claim_not_active" in str(exc.value)
    # The receipt is the wrong place to look: a result awaiting verification is
    # invisible there by design. The hint says so and names the way to finish.
    assert "awaiting verification" in exc.value.hint
    assert "--submission" in exc.value.hint


def test_a_second_auditor_is_superseded_not_failed(record):
    record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_first")
    second = record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_second")
    assert second.disposition == "superseded-pending-verification"
    assert second.superseded and second.recorded, "corroboration is not a failure"


def test_a_hash_that_contradicts_a_match_is_refused(record):
    """`match` plus a different hash is a contradiction; the API says submit
    `no-match` instead, which is a real outcome rather than a failure."""
    with pytest.raises(AuditError) as exc:
        record.submit(RUN, _bundle(Outcome.MATCH, hash_="b" * 64), claim="clm_h")
    assert "hash_contradicts_outcome" in str(exc.value)


# ── upload ───────────────────────────────────────────────────────────────────


def _handoff(tmp_path: Path, size: int = 200_000) -> Path:
    """A complete hand-off dir, plus the single-file bundle the record takes."""
    from conftest import write_handoff

    return write_handoff(tmp_path / "handoff", safetensors=size)


def _transport_bundle(tmp_path, size=200_000):
    """Exercise the low-level two-object transport, not handoff completeness."""
    path = _handoff(tmp_path, size) / "handoff.safetensors"
    return uploadmod.Bundle(path, uploadmod.digest_file(path), path.stat().st_size)


def test_the_bundle_digest_is_blake2b(tmp_path):
    """blake2b-256 of handoff.safetensors, per the contract: declared at submit
    time as artifactDigest and checked against the bytes after they land."""
    import hashlib

    src = _handoff(tmp_path)
    path = src / "handoff.safetensors"
    bundle = uploadmod.Bundle(path, uploadmod.digest_file(path), path.stat().st_size)
    assert bundle.path.name == "handoff.safetensors"
    assert (
        bundle.digest
        == hashlib.blake2b((src / "handoff.safetensors").read_bytes(), digest_size=32).hexdigest()
    )


def test_a_dcp_checkpoint_cannot_be_handed_off_and_says_why(tmp_path):
    """Name the missing validation without implying that the audit was lost."""
    from conftest import write_handoff

    d = write_handoff(tmp_path / "handoff")
    with pytest.raises(AuditError) as exc:
        uploadmod.build_bundle(d, tmp_path / "unused")
    assert "no packed handoff" in str(exc.value)
    assert "converter before submission" in exc.value.hint
    assert "still recorded" in exc.value.hint, "must not imply the audit was lost"


@pytest.mark.parametrize(
    "dropped", ["gradients.safetensors", "batch_hasher.rank_1.bin", "_COMPLETE"]
)
def test_an_incomplete_handoff_is_not_uploaded(tmp_path, dropped):
    """The recipient needs all of it. Discovering that after 18 GB helps nobody,
    and a partial artifact in intake is one the next auditor has to trust."""
    from conftest import write_handoff

    d = write_handoff(tmp_path / "handoff", world=2, safetensors=1000, omit=(dropped,))
    with pytest.raises(AuditError) as exc:
        uploadmod.build_bundle(d, tmp_path / "unused")
    assert "incomplete" in str(exc.value)
    assert dropped in exc.value.hint


def test_upload_accepts_the_packed_file_named_in_the_receipt(tmp_path):
    src = _handoff(tmp_path)
    path = src / "handoff.safetensors"
    bundle = uploadmod.build_bundle(path, tmp_path)
    assert bundle.path == path
    assert bundle.digest == uploadmod.digest_file(path)
    assert uploadmod.build_bundle(src, tmp_path) == bundle


def test_a_complete_handoff_carries_the_sidecars_the_recipient_needs(tmp_path):
    """What must travel, enumerated once so the uploader and the downloader
    cannot disagree about it."""
    from conftest import write_handoff

    from gensyn_audit import handoff as handoff_mod

    h = handoff_mod.inspect(write_handoff(tmp_path / "handoff", world=3))
    assert h.is_complete and h.has_gradients and h.dp_world_size == 3
    names = {p.name for p in h.files}
    assert {"gradients.safetensors", "batch_hasher.rank_2.bin", "meta.json"} <= names


def test_upload_needs_a_recorded_submission_first(record):
    with pytest.raises(AuditError, match="recorded submission"):
        record.upload_ticket(RUN, 200, claim="clm_nope", size=10, digest="d" * 64)


def test_upload_round_trips_and_the_record_checks_the_declared_digest(record, tmp_path):
    record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_up")
    bundle = _transport_bundle(tmp_path)
    ticket = record.upload_ticket(RUN, 200, claim="clm_up", size=bundle.size, digest=bundle.digest)
    uploadmod.send(bundle, ticket)
    assert record.verify_upload(200) is None


def _receipt(bundle, **over):
    """The fields of a submit.Result the sidecar is built from."""
    from types import SimpleNamespace

    fields = {
        "run": RUN,
        "audit_step": 200,
        "reproduced_hash": "a" * 64,
        "committed_hash": "a" * 64,
        "matched": True,
        "losses": [{"loss_ce": 2.5, "loss_zloss": 0.001}],
        "repop_commit": "244c0791e378",
        "repop_backends": ["metal"],
        "device": "mps",
        "tool": "gensyn-audit/0.1.0",
    }
    fields.update(over)
    return SimpleNamespace(**fields)


def test_the_sidecar_carries_exactly_what_the_verifier_checks(tmp_path):
    """The verifier requires eleven keys and rejects on a missing one. It
    compares step to the queue item's, which is the record's numbering, so
    audit_step goes in and until_step does not."""
    bundle = _transport_bundle(tmp_path)
    side = uploadmod.sidecar(_receipt(bundle), bundle, run_id="20260722-213626-ad3276b")
    assert set(side) == {
        "run_id",
        "step",
        "state_hash",
        "expected_state_hash",
        "match",
        "ce",
        "z_loss",
        "bundle_digest",
        "repop",
        "device",
        "produced_by",
    }
    # The verifier expects its configured run id, not the API's addressable name.
    assert side["run_id"] == "20260722-213626-ad3276b"
    assert uploadmod.sidecar(_receipt(bundle), bundle)["run_id"] == RUN, "no id known: as addressed"
    assert side["step"] == 200
    assert side["match"] is True
    assert side["bundle_digest"] == bundle.digest
    assert side["ce"] == 2.5 and side["z_loss"] == 0.001
    assert side["repop"] == {"commit": "244c0791e378", "backends": ["metal"]}


def test_the_sidecar_is_uploaded_beside_the_bundle(record, tmp_path):
    record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_side")
    bundle = _transport_bundle(tmp_path)
    ticket = record.upload_ticket(
        RUN, 200, claim="clm_side", size=bundle.size, digest=bundle.digest
    )
    uploadmod.send(bundle, ticket)
    assert record.verify_sidecar(200) is not None, "nothing sent yet"
    uploadmod.send_sidecar(uploadmod.sidecar(_receipt(bundle), bundle), ticket)
    assert record.verify_sidecar(200) is None


def test_a_record_without_a_sidecar_url_is_refused_rather_than_half_uploaded(tmp_path):
    from gensyn_audit.record import UploadTicket

    bundle = _transport_bundle(tmp_path)
    with pytest.raises(AuditError, match="no URL for handoff.json"):
        uploadmod.send_sidecar(
            uploadmod.sidecar(_receipt(bundle), bundle), UploadTicket(signed_url="file:///dev/null")
        )


def test_an_interrupted_upload_resumes_instead_of_restarting(record, tmp_path):
    """16-18 GB on residential upstream takes hours; restarting from zero on
    every drop is how a volunteer gives up."""
    record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_res")
    bundle = _transport_bundle(tmp_path, 500_000)
    ticket = record.upload_ticket(RUN, 200, claim="clm_res", size=bundle.size, digest=bundle.digest)

    target = Path(ticket.signed_url[len("file://") :])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(bundle.path.read_bytes()[: bundle.size // 2])

    sent: list[int] = []
    uploadmod.send(bundle, ticket, on_progress=lambda done, total: sent.append(done))
    assert record.verify_upload(200) is None, "resumed bytes must reassemble exactly"
    assert sent and sent[0] > bundle.size // 2 - 1, "must resume, not restart"


def test_a_complete_upload_is_not_resent(record, tmp_path):
    record.submit(RUN, _bundle(Outcome.MATCH), claim="clm_done")
    bundle = _transport_bundle(tmp_path, 50_000)
    ticket = record.upload_ticket(
        RUN, 200, claim="clm_done", size=bundle.size, digest=bundle.digest
    )
    uploadmod.send(bundle, ticket)
    before = Path(ticket.signed_url[len("file://") :]).stat().st_size
    uploadmod.send(bundle, ticket)
    assert Path(ticket.signed_url[len("file://") :]).stat().st_size == before


# ── the mock must never pass for real ────────────────────────────────────────


def test_the_mock_announces_itself(record):
    assert record.is_mock
    assert record.label.startswith("mock://")
    assert "MOCK" in MockRecord.banner()
    prov = record.provenance()
    assert prov["record"] == "mock"
    assert "No real service was contacted" in prov["note"]


# ── a record that admits it is not real ──────────────────────────────────────


def _manifest(**over) -> dict:
    doc = {
        "run": {
            "id": "20260623-191118-553c06d",
            "name": "open-1b",
            "totalSteps": 80957,
            "stepsPerSegment": 100,
        },
        "audit": {
            "artifactCheck": "loss-checked",
            "claimTtlHours": 48,
            "exchangeFormat": "safetensors",
        },
        "endpoints": {"receipt": "/v1/runs/open-1b/steps/{step}/receipt"},
        "simulated": False,
        "simulatedParts": {},
    }
    doc.update(over)
    return doc


def test_the_manifest_names_the_run_by_name_not_id():
    """The endpoint index addresses `/v1/runs/open-1b/…`, so the name is what
    the record publishes as its address."""
    from gensyn_audit.record import _parse_manifest

    m = _parse_manifest(_manifest())
    assert m.run_name == "open-1b"
    assert m.run_id == "20260623-191118-553c06d"
    assert m.artifact_check == "loss-checked"
    assert m.exchange_format == "safetensors"


def test_simulated_parts_are_read_individually():
    """Staging derives placeholder commitments AND has no verification service.
    They are separate admissions and a record could make either one alone."""
    from gensyn_audit.record import _parse_manifest

    live = _parse_manifest(_manifest())
    assert not live.simulated and live.loss_gate_is_live
    assert not live.commitments_are_placeholders

    staging = _parse_manifest(
        _manifest(
            simulated=True,
            simulatedParts={
                "commitments": "derived placeholders",
                "lossGate": "not wired up",
            },
        )
    )
    assert staging.simulated
    assert staging.commitments_are_placeholders
    assert not staging.loss_gate_is_live


def test_an_interval_against_a_simulated_record_is_refused(monkeypatch):
    """Twelve hours against a hash the run never produced is the worst outcome
    this tool can deliver, so it blocks rather than warns."""
    import argparse

    from gensyn_audit import cli
    from gensyn_audit.record import _parse_manifest

    class FakeRecord:
        is_mock = False

        @staticmethod
        def manifest():
            return _parse_manifest(
                _manifest(simulated=True, simulatedParts={"commitments": "derived placeholders"})
            )

    args = argparse.Namespace(step=25701, allow_simulated=False)
    with pytest.raises(AuditError, match="placeholder commitments"):
        cli._warn_if_simulated(FakeRecord(), args)

    # Explicitly opting in is allowed: exercising the plumbing is legitimate.
    args.allow_simulated = True
    cli._warn_if_simulated(FakeRecord(), args)


def test_an_init_unit_is_not_blocked_by_a_simulated_record(monkeypatch):
    """It is verified locally against the kit's trajectory; the record plays no
    part, so its state is irrelevant."""
    import argparse

    from gensyn_audit import cli
    from gensyn_audit.record import _parse_manifest

    class FakeRecord:
        is_mock = False

        @staticmethod
        def manifest():
            return _parse_manifest(_manifest(simulated=True, simulatedParts={"commitments": "x"}))

    cli._warn_if_simulated(FakeRecord(), argparse.Namespace(step=None, allow_simulated=False))


def test_the_ledger_is_jsonl_not_json():
    """One `json.loads` over the whole body fails on line two — which is how
    this path stayed broken until something actually called it."""
    from gensyn_audit.record import HttpRecord

    rec = HttpRecord("https://example.invalid")
    rec._request_text = lambda m, p, **kw: (
        '{"step": 0, "state": "confirmed"}\n'
        '{"step": 1, "state": "provisional"}\n'
        '{"step": 2, "truncated'  # a partial tail
    )
    rows = rec.ledger(limit=3)
    assert [r["step"] for r in rows] == [0, 1], "a truncated tail is skipped, not fatal"


def test_an_upload_without_a_submission_id_is_refused():
    """The upload must explicitly identify the submission receiving its bytes."""
    from gensyn_audit.record import HttpRecord

    rec = HttpRecord("https://example.invalid")
    with pytest.raises(AuditError, match="submissionId"):
        rec.upload_ticket("run", 1, claim="clm", size=1, digest="d" * 64)
