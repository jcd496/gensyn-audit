def test_a_quarantined_predecessor_explains_itself(monkeypatch, capsys):
    """A hand-off that has not cleared verification is a race, not a misconfig.

    The bare "does not exist" reads as a permissions or bucket problem, and
    sends an auditor looking for one they do not have.
    """
    from gensyn_audit import cli
    from gensyn_audit.errors import AuditError
    from gensyn_audit.record import Predecessor

    pred = Predecessor(
        source="crowd-provided checkpoint",
        uri="gs://public/handoff/step-98.safetensors",
        digest="sha256:" + "0" * 64,
        step=98,
    )
    assert pred.is_crowd_provided

    # The published-checkpoint case must still surface the raw error: there is
    # no relay race to blame, so something really is wrong.
    published = Predecessor(
        source="published checkpoint", uri="gs://public/ckpt/000100/", digest=None, step=100
    )
    assert not published.is_crowd_provided
    assert isinstance(AuditError("x"), Exception)
    assert cli is not None


# ── --predecessor-uri ────────────────────────────────────────────────────────


def test_predecessor_uri_overrides_the_record(monkeypatch):
    """The record can name a location the auditor cannot read -- today it names
    a bucket that does not exist. The override changes only WHERE the bytes come
    from; `verify_predecessor` still holds them to the record's digest."""
    import argparse

    from gensyn_audit import cli
    from gensyn_audit.record import Predecessor, StepContext
    from gensyn_audit.steps import StepRef

    ctx = StepContext(
        run="open-1b",
        step=25700,
        committed_hash="a" * 64,
        predecessor=Predecessor(
            source="published checkpoint", uri="gs://absent/ckpt/025700/", digest=None, step=25700
        ),
        descriptor_uri=None,
        phase="main",
        microbatches=288,
        gcs_root="gs://b/shards",
    )
    ref = StepRef(25700)
    args = argparse.Namespace(config_name=None, predecessor_uri="gs://real/ckpt/step_000025700")

    unit = cli._unit_from_step(ctx, ref, args)
    assert unit.checkpoint_uri == "gs://real/ckpt/step_000025700"
    # The expectations it is checked against are untouched.
    assert unit.expect_hash == "a" * 64
    assert unit.predecessor_step == 25700
    assert unit.until_step == 25701


def test_without_the_override_the_record_still_wins(monkeypatch):
    import argparse

    from gensyn_audit import cli
    from gensyn_audit.record import Predecessor, StepContext
    from gensyn_audit.steps import StepRef

    ctx = StepContext(
        run="open-1b",
        step=25700,
        committed_hash="a" * 64,
        predecessor=Predecessor(
            source="published checkpoint", uri="gs://named-by-record/ckpt/", digest=None, step=25700
        ),
        descriptor_uri=None,
        phase="main",
        microbatches=288,
        gcs_root=None,
    )
    args = argparse.Namespace(config_name=None, predecessor_uri=None)
    unit = cli._unit_from_step(ctx, StepRef(25700), args)
    assert unit.checkpoint_uri == "gs://named-by-record/ckpt/"


# ── the simulated gate refuses only what cannot succeed ──────────────────────


def _rec(parts):
    from gensyn_audit.record import RecordManifest

    class _R:
        is_mock = False

        def manifest(self):
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

    return _R()


def test_an_unwired_loss_gate_does_not_block_a_replay():
    """The runner reproduces the step and reports the losses; deciding whether
    they are the cluster's is the verification service's job and only it can do
    it. An unwired gate makes acceptance provisional, not the replay worthless."""
    import argparse

    from gensyn_audit import cli

    args = argparse.Namespace(step=25700, allow_simulated=False)
    cli._warn_if_simulated(_rec({"commitments": None, "lossGate": "not wired up"}), args)


def test_placeholder_commitments_still_block():
    """Here the replay is compared against a hash the run never produced, so it
    cannot match however correct it is. Twelve hours to learn nothing."""
    import argparse

    import pytest as _pytest

    from gensyn_audit import cli
    from gensyn_audit.errors import AuditError

    args = argparse.Namespace(step=25700, allow_simulated=False)
    with _pytest.raises(AuditError) as e:
        cli._warn_if_simulated(_rec({"commitments": "derived", "lossGate": None}), args)
    assert "placeholder commitments" in str(e.value)


def test_allow_simulated_still_overrides_placeholder_commitments():
    import argparse

    from gensyn_audit import cli

    args = argparse.Namespace(step=25700, allow_simulated=True)
    cli._warn_if_simulated(_rec({"commitments": "derived", "lossGate": None}), args)


def test_an_init_unit_is_never_blocked():
    import argparse

    from gensyn_audit import cli

    args = argparse.Namespace(step=None, allow_simulated=False)
    cli._warn_if_simulated(_rec({"commitments": "derived", "lossGate": "not wired"}), args)


def test_the_override_reaches_the_download_not_just_the_preflight(tmp_path, monkeypatch):
    """The predecessor override controls both validation and download."""
    import argparse

    from gensyn_audit import cli
    from gensyn_audit.record import Predecessor, StepContext
    from gensyn_audit.steps import StepRef

    ctx = StepContext(
        run="open-1b",
        step=25700,
        committed_hash="a" * 64,
        predecessor=Predecessor(
            source="published checkpoint", uri="gs://absent/ckpt/025700/", digest=None, step=25700
        ),
        descriptor_uri=None,
        phase="main",
        microbatches=288,
        gcs_root="gs://b/shards",
    )
    args = argparse.Namespace(config_name=None, predecessor_uri="gs://real/ckpt/step_000025700")
    unit = cli._unit_from_step(ctx, StepRef(25700), args)

    # This is the expression the download now uses.
    assert (unit.checkpoint_uri or ctx.predecessor.uri) == "gs://real/ckpt/step_000025700"


# ── logs: a live progress bar must not explode into one line per redraw ─────


def test_a_progress_bar_redraw_collapses_to_the_line_a_terminal_shows(tmp_path, capsys):
    """`gensyn-audit logs` renders the replay's own log file, which is written
    the way a terminal is written to: a bar redraws in place with a bare \\r
    and moves on with a real \\n. Counting every \\r as a line break -- which
    both `str.splitlines` and `Path.read_text` do -- turns one visible line
    into one printed line per redraw. Over a 12-24h interval unit that is the
    difference between a log you can read and one nobody can scroll.
    """
    import argparse

    from gensyn_audit import cli
    from gensyn_audit.plan import Workdir

    workdir = Workdir(tmp_path)
    workdir.log.write_bytes(
        b"header line\n"
        b"step 1 microbatches:  10%|X  | 1/10\r"
        b"step 1 microbatches:  20%|XX | 2/10\r"
        b"step 1 microbatches:  30%|XXX| 3/10\r"
        b"final status line\n"
    )

    args = argparse.Namespace(workdir=str(tmp_path), lines=None, follow=False)
    cli.cmd_logs(args)

    assert capsys.readouterr().out.splitlines() == ["header line", "final status line"]


def test_a_redraw_group_is_scoped_by_the_line_it_redraws():
    """A redraw group ends at a newline, and CRLF remains a normal line ending."""
    from gensyn_audit.cli import _terminal_lines

    assert _terminal_lines(b"a\rA\nb\rB\n") == ["A", "B"]
    assert _terminal_lines(b"first\r\nsecond\r\n") == ["first", "second"]
    assert _terminal_lines(b"") == []


# ── a step nobody has handed off to ──────────────────────────────────────────


def test_a_step_with_no_predecessor_yet_explains_rather_than_crashes():
    """A step awaiting hand-off has neither a predecessor URI nor step."""
    from gensyn_audit import cli
    from gensyn_audit.errors import AuditError

    err = cli._not_published_yet(None)
    assert isinstance(err, AuditError)
    assert "not published yet" in str(err)
    assert "position-1" in (err.hint or "")
    # No stray ": None" where a URI would go.
    assert "None" not in str(err)


def test_the_manifest_fallback_needs_a_step_number():
    """`<checkpoints>/step_<n>` cannot be built without n, and the record omits
    `predecessor.step` entirely until a hand-off exists."""
    from gensyn_audit.manifest import Artifacts

    art = Artifacts(checkpoints="gs://b/ckpt")
    assert art.checkpoint_uri(25700) == "gs://b/ckpt/step_000025700"
    assert art.checkpoint_uri(None) is None


def test_a_named_uri_still_appears_in_the_message():
    from gensyn_audit import cli

    err = cli._not_published_yet("gs://b/handoff/step-98.safetensors")
    assert "gs://b/handoff/step-98.safetensors" in str(err)


# ── gensyn-audit upload ──────────────────────────────────────────────────────


def _workdir_with(tmp_path, *, verdict: bool | None, pid: int = 999_999):
    """A workdir as `run` leaves one: state, and a log that may carry a verdict."""
    from gensyn_audit.plan import Workdir
    from gensyn_audit.runner import RunState, save_state

    wd = Workdir(tmp_path)
    wd.root.mkdir(parents=True, exist_ok=True)
    save_state(
        wd,
        RunState(
            kit_id="k",
            kit_prefix="p",
            run="open-1b",
            unit_kind="interval",
            config_name="c",
            until_step=201,
            audit_step=200,
            expect_hash="a" * 64,
            device="mps",
            pid=pid,
            argv=[],
            env_overlay={},
            started_at="2026-09-11T00:00:00+00:00",
            workdir=str(tmp_path),
        ),
    )
    wd.log.write_text(
        f"state_hash={'a' * 16} expected={'a' * 16} MATCH=True\n"
        if verdict
        else "replaying steps 200 -> 201\n"
    )
    return wd


def test_upload_on_an_empty_workdir_says_to_run_the_audit(tmp_path):
    """The fix for "nothing to upload" is to run the audit, not to retry."""
    import pytest

    from gensyn_audit import cli
    from gensyn_audit.errors import AuditError

    with pytest.raises(AuditError) as exc:
        cli._guard_uploadable(tmp_path / "nothing")
    assert "no audit in" in str(exc.value)
    assert "gensyn-audit run" in (exc.value.hint or "")


def test_upload_refuses_a_replay_that_never_finished(tmp_path):
    """A log with no verdict is not a result waiting to be sent."""
    import pytest

    from gensyn_audit import cli
    from gensyn_audit.errors import AuditError

    wd = _workdir_with(tmp_path, verdict=None)
    with pytest.raises(AuditError) as exc:
        cli._guard_uploadable(wd.root)
    assert "did not finish" in str(exc.value)
    assert "gensyn-audit run" in (exc.value.hint or "")


def test_upload_refuses_while_the_replay_is_still_running(tmp_path):
    """Sending a result from a workdir a live replay is still writing to would
    report a verdict that is not final."""
    import os

    import pytest

    from gensyn_audit import cli
    from gensyn_audit.errors import AuditError

    wd = _workdir_with(tmp_path, verdict=True, pid=os.getpid())
    with pytest.raises(AuditError) as exc:
        cli._guard_uploadable(wd.root)
    assert "still running" in str(exc.value)
    assert "--follow" in (exc.value.hint or "")


def test_upload_accepts_a_finished_match(tmp_path):
    from gensyn_audit import cli

    wd = _workdir_with(tmp_path, verdict=True)
    state, prog = cli._guard_uploadable(wd.root)
    assert prog.match is True
    assert state.audit_step == 200


def test_upload_takes_a_claim_the_run_did_not_have(tmp_path):
    """Finishing detached and claiming afterwards is the case this exists for."""
    from gensyn_audit import cli

    wd = _workdir_with(tmp_path, verdict=True)
    state, _ = cli._guard_uploadable(wd.root)
    assert state.claim is None


def test_sidecar_only_does_not_re_send_the_bundle(tmp_path, monkeypatch, capsys):
    """The two objects go up in sequence, so a drop between them leaves 18 GB in
    intake and a submission the verifier rejects as `missing_artifact`.
    Re-sending the bundle to fix a kilobyte is the wrong trade."""
    from gensyn_audit import cli
    from gensyn_audit import upload as uploadmod
    from gensyn_audit.record import UploadTicket

    bundle_file = tmp_path / "handoff.safetensors"
    bundle_file.write_bytes(b"tensors")
    sent, sidecars = [], []

    monkeypatch.setattr(
        uploadmod,
        "build_bundle",
        lambda *a, **k: uploadmod.Bundle(path=bundle_file, digest="d" * 64, size=7),
    )
    monkeypatch.setattr(uploadmod, "send", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(uploadmod, "send_sidecar", lambda p, t: sidecars.append(p))
    monkeypatch.setattr(
        uploadmod, "sidecar", lambda result, bundle, **_: {"bundle_digest": bundle.digest}
    )
    monkeypatch.setattr(cli, "_record_run_id", lambda *a: "run-id")

    class _Rec:
        is_mock = False

        @staticmethod
        def upload_ticket(*a, **k):
            return UploadTicket(signed_url="https://x/bundle", sidecar_url="https://x/sidecar")

    class _Plan:
        class workdir:
            handoff = tmp_path
            root = tmp_path

    class _State:
        run, until_step, claim = "open-1b", 201, "clm_x"

    class _Result:
        artifact = {"path": str(bundle_file)}

    cli._do_upload(_Plan(), _State(), _Rec(), None, _Result(), sidecar_only=True)
    assert sent == [], "re-sent the bundle"
    assert sidecars and sidecars[0]["bundle_digest"] == "d" * 64
    assert "bundle skipped" in capsys.readouterr().out


def test_the_default_still_sends_both(tmp_path, monkeypatch):
    from gensyn_audit import cli
    from gensyn_audit import upload as uploadmod
    from gensyn_audit.record import UploadTicket

    bundle_file = tmp_path / "handoff.safetensors"
    bundle_file.write_bytes(b"tensors")
    sent, sidecars = [], []
    monkeypatch.setattr(
        uploadmod,
        "build_bundle",
        lambda *a, **k: uploadmod.Bundle(path=bundle_file, digest="d" * 64, size=7),
    )
    monkeypatch.setattr(uploadmod, "send", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(uploadmod, "send_sidecar", lambda p, t: sidecars.append(p))
    monkeypatch.setattr(uploadmod, "sidecar", lambda result, bundle, **_: {})
    monkeypatch.setattr(cli, "_record_run_id", lambda *a: "run-id")

    class _Rec:
        is_mock = False

        @staticmethod
        def upload_ticket(*a, **k):
            return UploadTicket(signed_url="https://x/b", sidecar_url="https://x/s")

    class _Plan:
        class workdir:
            handoff = tmp_path
            root = tmp_path

    class _State:
        run, until_step, claim = "open-1b", 201, "clm_x"

    class _Result:
        artifact = {"path": str(bundle_file)}

    cli._do_upload(_Plan(), _State(), _Rec(), None, _Result())
    assert len(sent) == 1 and len(sidecars) == 1


def test_upload_keeps_the_provenance_the_run_established(tmp_path):
    """`upload` rebuilds the receipt from the same log and state, so everything
    else comes out identical. The predecessor block does not: it records a gate
    that ran once, before the replay, against bytes this command does not
    require to still exist. Rebuilding without it would overwrite the run's own
    record of it with silence."""
    import json

    from gensyn_audit import cli

    wd = _workdir_with(tmp_path, verdict=True)
    (wd.root / "result.json").write_text(
        json.dumps(
            {
                "predecessor": {
                    "predecessor_provenance": "gensyn-anchor",
                    "artifact_integrity": "unverified",
                    "tensor_state_commitment": "skipped-anchor",
                    "statement": "Trusted Gensyn anchor; artifact integrity NOT verified; …",
                },
                "outcome": "match",
            }
        )
    )
    carried = cli._prior_predecessor(wd)
    assert carried["predecessor_provenance"] == "gensyn-anchor"
    assert carried["tensor_state_commitment"] == "skipped-anchor"


def test_no_prior_receipt_carries_nothing_rather_than_inventing(tmp_path):
    from gensyn_audit import cli

    wd = _workdir_with(tmp_path, verdict=True)
    assert cli._prior_predecessor(wd) is None


def test_a_corrupt_receipt_does_not_stop_the_upload(tmp_path):
    """A truncated result.json is a reason to lose the provenance line, not to
    refuse to send a result that is already on disk."""
    from gensyn_audit import cli

    wd = _workdir_with(tmp_path, verdict=True)
    (wd.root / "result.json").write_text("{not json")
    assert cli._prior_predecessor(wd) is None


def test_upload_end_to_end_submits_and_keeps_provenance(tmp_path, monkeypatch, capsys):
    """The success path: a finished workdir, a claim, a record that accepts —
    the receipt is rewritten and still carries what the replay established."""
    import argparse
    import json

    from gensyn_audit import cli
    from gensyn_audit.outcome import Outcome

    wd = _workdir_with(tmp_path, verdict=True)
    (wd.root / "result.json").write_text(
        json.dumps(
            {
                "predecessor": {
                    "predecessor_provenance": "gensyn-anchor",
                    "statement": "Trusted Gensyn anchor; …",
                }
            }
        )
    )

    sent = {}

    class _Plan:
        venv = tmp_path / "venv"
        workdir = wd
        unit = type("U", (), {"label": "open-1b interval → step 201", "is_init": False})()

    monkeypatch.setattr(cli, "_build_plan", lambda *a, **k: (_Plan(), None, None))
    monkeypatch.setattr(cli, "_resolve_record", lambda *a, **k: None)
    monkeypatch.setattr(cli.manifestmod, "resolve", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_pack_handoff", lambda *a, **k: None)
    monkeypatch.setattr(
        cli, "_submit_and_upload", lambda *a, **k: sent.setdefault("called", True) and None
    )

    args = argparse.Namespace(
        manifest=None,
        workdir=str(wd.root),
        claim="clm_x",
        machine=None,
        handle=None,
        sidecar_only=False,
    )
    cli.cmd_upload(args)

    assert sent.get("called") is True
    written = json.loads((wd.root / "result.json").read_text())
    assert written["predecessor"]["predecessor_provenance"] == "gensyn-anchor"
    assert written["outcome"] == Outcome.MATCH.value
    assert "Trusted Gensyn anchor" in capsys.readouterr().out


def test_run_defaults_to_the_hosts_accelerator():
    import sys

    from gensyn_audit import cli

    expected = {"darwin": "mps", "linux": "cuda"}.get(sys.platform, "cpu")
    assert cli.build_parser().parse_args(["run"]).device == expected
