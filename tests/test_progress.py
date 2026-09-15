"""The log parser, against the shapes audit_replay actually emits."""

from gensyn_audit import progress

GOOD = "2e6f724bd1f1d9ae88d59fdc9831c5b46478f5ddb902ffe1e6da9162cd98ab7b"


def test_match_reports_the_full_digest_not_the_logged_one():
    """The MATCH line truncates to 16 chars; reporting that as the hash is wrong."""
    log = f"""
2026-08-27 10:00:00,000 INFO pretrain.cli.audit_replay :: replaying steps 800 → 801
2026-08-27 10:00:01,000 INFO pretrain.cli.audit_replay :: state_hash={GOOD[:16]} expected={GOOD[:16]} MATCH=True
{{
  "step": 801,
  "state_hash": "{GOOD}",
  "expected": "{GOOD}",
  "match": true
}}
"""
    p = progress.parse(log)
    assert p.match is True
    assert p.phase == "done"
    assert p.state_hash == GOOD, "full digest must come from the result object"
    assert p.state_hash_short == GOOD[:16]
    assert len(p.state_hash) == 64


def test_mismatch_takes_both_full_digests_from_the_failure_line():
    """On the sad path there is no JSON -- SystemExit is the only full source."""
    got = "a" * 64
    log = (
        f"2026-08-27 10:00:01,000 INFO pretrain.cli.audit_replay :: "
        f"state_hash={got[:16]} expected={GOOD[:16]} MATCH=False\n"
        f"AUDIT FAILED: state_hash {got} != {GOOD}\n"
    )
    p = progress.parse(log)
    assert p.match is False
    assert p.phase == "failed"
    assert p.state_hash == got
    assert p.expected_hash == GOOD


def test_reads_the_newest_tqdm_redraw_not_the_last_line():
    """tqdm never terminates a line; the newest bar is the last \\r field."""
    log = (
        "2026-08-27 10:00:00,000 INFO pretrain.cli.audit_replay :: replaying steps 800 → 801\n"
        "\r  step 801 microbatches:   3%|#    | 9/288 [00:30<15:00,  3.2s/mb]"
        "\r  step 801 microbatches:  42%|#### | 121/288 [1:23:45<3:12:01,  3.2s/mb]"
    )
    p = progress.parse(log)
    assert (p.microbatches_done, p.microbatches_total) == (121, 288)
    assert p.bar_eta == "3:12:01"
    assert abs(p.fraction - 121 / 288) < 1e-9
    assert p.phase == "replaying"


def test_memlog_is_picked_up_for_the_swap_signal():
    log = (
        "2026-08-27 10:00:00,000 INFO pretrain.cli.audit_replay :: "
        "MEMLOG[post-step] host_RSS=41.9GB mps_alloc=22.0GB mps_driver=43.2GB\n"
    )
    p = progress.parse(log)
    assert p.host_rss_gb == 41.9
    assert p.mps_driver_gb == 43.2


def test_memlog_without_an_accelerator_still_parses():
    log = "2026-08-27 10:00:00,000 INFO pretrain.cli.audit_replay :: MEMLOG[start] host_RSS=3.5GB\n"
    p = progress.parse(log)
    assert p.host_rss_gb == 3.5
    assert p.mps_driver_gb is None


def test_spike_halt_is_surfaced_as_reproduction_not_failure():
    log = (
        "2026-08-27 10:00:00,000 WARNING pretrain.cli.audit_replay :: "
        "spike protocol halt at step 801 during audit: threshold\n"
    )
    assert progress.parse(log).halted_at_step == 801


def test_handoff_checkpoint_is_captured():
    log = (
        "2026-08-27 10:00:00,000 INFO pretrain.cli.audit_replay :: saved chained-audit "
        "checkpoint → /tmp/handoff/step_801 (step=801 consumed_tokens=4096 "
        "chained_hash=2e6f724bd1f1d9ae); audit the next interval with "
        "--checkpoint /tmp/handoff/step_801\n"
    )
    assert progress.parse(log).saved_checkpoint == "/tmp/handoff/step_801"


def test_empty_log_is_not_a_verdict():
    p = progress.parse("")
    assert p.match is None and not p.finished and p.fraction is None


# ── the result object is now the source of provenance and losses ─────────────


def test_provenance_and_losses_come_from_the_result_not_the_log():
    """audit_replay records the kernel build in every result. Scraping it out
    of log prose would be guessing at the one field that makes a hash evidence."""
    log = """
2026-09-07 10:00:00,000 INFO pretrain.audit :: replaying steps 800 → 801
{
  "step": 801,
  "consumed_tokens": 4096,
  "state_hash": "__H__",
  "repop": {"commit": "244c0791e378a180dd6b29bbf8f2e244b2bea806",
            "backends": ["cpu", "metal"], "cuda_arch_list": ""},
  "device": "mps",
  "rank0_losses": [{"step": 801, "consumed_tokens": 4096,
                    "loss_ce": 2.718281828459045, "loss_zloss": 9.99e-05}],
  "expected": "__H__",
  "match": true
}
""".replace("__H__", GOOD)
    p = progress.parse(log)
    assert p.repop_commit == "244c0791e378a180dd6b29bbf8f2e244b2bea806"
    assert p.repop_backends == ("cpu", "metal")
    assert p.device == "mps"
    assert p.rank0_losses[-1]["loss_ce"] == 2.718281828459045


def test_init_mode_is_recognised():
    log = (
        '{\n  "step": 0,\n  "state_hash": "__H__",\n  "mode": "init",\n  "match": true\n}\n'
    ).replace("__H__", GOOD)
    assert progress.parse(log).mode == "init"


def test_loss_log_is_read_from_its_own_file(tmp_path):
    """It is rewritten atomically each step, so it is the only per-step source
    that survives a killed process."""
    import json

    path = tmp_path / "losses.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {"step": 801, "consumed_tokens": 4096, "loss_ce": 2.5, "loss_zloss": 1e-4}
                ]
            }
        )
    )
    assert progress.read_loss_log(path)[-1]["step"] == 801
    assert progress.read_loss_log(tmp_path / "absent.json") == []


def test_a_torn_loss_log_reads_as_empty_not_a_crash(tmp_path):
    path = tmp_path / "losses.json"
    path.write_text('{"records": [{"step": 8')
    assert progress.read_loss_log(path) == []


# ── against real tqdm output ─────────────────────────────────────────────────

#: Captured by running audit_replay's own micro-batch bar construction
#: (`tqdm(total=None, desc="  microbatches", unit="mb", dynamic_ncols=True,
#: position=1, leave=False)` then `.reset(total=M)`, `.set_description(...)`,
#: `.update(1)`) under the kit's tqdm with its output redirected to a PIPE —
#: exactly how the runner captures it. Pinned as bytes rather than hand-written
#: so the regex is tested against what tqdm really emits, not what I assumed.
_REAL_TQDM = (
    "  microbatches: 0mb [00:00, ?mb/s]\x1b[A\r"
    "  microbatches:   0%|          | 0/288 [00:00<?, ?mb/s]\x1b[A\r"
    "  step 200 microbatches:   0%|          | 0/288 [00:00<?, ?mb/s]\x1b[A\r"
    "  step 200 microbatches:  17%|█▋        | 48/288 [00:12<01:02, 3.85mb/s]\x1b[A\r"
    "  step 200 microbatches:  50%|█████     | 144/288 [01:23<01:24, 1.71mb/s]\x1b[A\r"
)


def test_reads_the_bar_tqdm_actually_writes_to_a_pipe():
    """tqdm does not disable itself on a non-TTY, so an interval replay's
    micro-batch counts do reach us through the pipe. (An init unit produces no
    bar at all: it returns before audit_replay ever constructs one.)"""
    p = progress.parse(_REAL_TQDM)
    assert (p.microbatches_done, p.microbatches_total) == (144, 288)
    assert p.fraction == 0.5
    assert p.bar_eta == "01:24"
    assert p.bar_elapsed == "01:23"


def test_the_trailing_clear_does_not_erase_the_last_known_position():
    """`leave=False` blanks the line when a step ends. The last real reading
    must survive that, or the display would drop to nothing at each step."""
    p = progress.parse(_REAL_TQDM + " " * 75 + "\x1b[A\r")
    assert (p.microbatches_done, p.microbatches_total) == (144, 288)


def test_the_outer_step_bar_is_not_mistaken_for_microbatches():
    """audit_replay runs two bars; only one counts micro-batches."""
    outer = "audit replay:  40%|████      | 2/5 [10:00<15:00, 300s/it]\r"
    p = progress.parse(outer + _REAL_TQDM)
    assert p.microbatches_total == 288, "must not latch onto the step bar"


# The mismatch path, taken verbatim from the first tester report (2026-09-13,
# OPEN-1B audit step 103). audit_replay logs the digest truncated to 16 hex and
# then raises, so its success-path JSON block never prints -- which is exactly
# the case the CLI used to render as `reproduced ?`.
_MISMATCH_LOG = (
    "2026-09-13 08:03:08,182 INFO pretrain.audit :: "
    "state_hash=2c6c650b09949ad8 expected=e65f46aecf3ceaf1 MATCH=False\n"
    "Traceback (most recent call last):\n"
    "ValueError: state_hash did NOT match --expect-hash; refusing to save a "
    "checkpoint that would chain the next audit off an unreproduced state.\n"
)


def test_mismatch_without_a_result_block_still_reports_the_digest():
    p = progress.parse(_MISMATCH_LOG)
    assert p.match is False
    # The full digest is genuinely unavailable here; the prefix is not.
    assert p.state_hash is None
    assert p.reproduced == "2c6c650b09949ad8"
    assert p.reproduced_is_truncated is True


def test_a_matching_result_block_reports_the_full_digest_untruncated():
    full = "6fd7588d9dc8535662b9b80da7e00f2cac19c0445574f351fd522e30bac80692"
    log = (
        "2026-09-12 10:11:19,821 INFO pretrain.audit :: "
        f"state_hash={full[:16]} expected={full[:16]} MATCH=True\n"
        "{\n"
        f'  "state_hash": "{full}",\n'
        f'  "expected": "{full}",\n'
        '  "match": true\n'
        "}\n"
    )
    p = progress.parse(log)
    assert p.match is True
    assert p.reproduced == full
    assert p.reproduced_is_truncated is False


def test_no_digest_at_all_is_not_reported_as_one():
    p = progress.parse("2026-09-13 08:00:00,000 INFO pretrain.audit :: starting\n")
    assert p.reproduced is None and p.reproduced_is_truncated is False
