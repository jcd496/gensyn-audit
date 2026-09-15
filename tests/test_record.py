# ── simulatedParts uses null, not absence ────────────────────────────────────


def _manifest(parts):
    from gensyn_audit.record import RecordManifest

    return RecordManifest(
        run_id="r",
        run_name="open-1b",
        total_steps=None,
        steps_per_segment=None,
        artifact_check=None,
        claim_ttl_hours=None,
        exchange_format=None,
        simulated=True,
        simulated_parts=parts,
        endpoints={},
    )


def test_null_part_means_that_part_is_real():
    # The record lists every part as a key: a string explains what is simulated,
    # null means it is genuine. Testing for the key alone fired the loudest
    # warning this tool has against a real committed hash.
    m = _manifest({"commitments": None, "lossGate": "not wired up"})
    assert m.commitments_are_placeholders is False
    assert m.loss_gate_is_live is False


def test_a_message_means_that_part_is_simulated():
    m = _manifest({"commitments": "derived placeholders", "lossGate": None})
    assert m.commitments_are_placeholders is True
    assert m.loss_gate_is_live is True


def test_the_codes_the_record_actually_returns_are_explained():
    """`no_token` and `bad_request` are both live on production and were
    surfacing as bare codes. The first almost always means a missing --claim."""
    from gensyn_audit.record import _ERROR_GUIDANCE

    assert "--claim" in _ERROR_GUIDANCE["no_token"]
    assert "not your" in _ERROR_GUIDANCE["bad_request"]
