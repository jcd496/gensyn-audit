"""The resumable upload against a fake GCS: which URL gets asked what.

The first production upload died on `could not query the upload's progress
(400)`: the progress query went to the signed *start* URL, whose signature is
over `x-goog-resumable` on a POST, and GCS answered MalformedSecurityHeader.
Only the session URI from `Location` answers a status query. These tests hold
the fake to the same rule, and check that a re-run resumes the session it
opened rather than opening another and starting from byte 0.
"""

from __future__ import annotations

import email.message
import io
import json
import urllib.error
import urllib.request

import pytest

from gensyn_audit import upload as uploadmod
from gensyn_audit.errors import AuditError
from gensyn_audit.record import UploadTicket

SIGNED = "https://storage.example/intake/uploads/sub_1/handoff.safetensors?X-Goog-Signature=abc"
OBJECT = "gs://intake/uploads/sub_1/handoff.safetensors"


def _headers(**kv) -> email.message.Message:
    h = email.message.Message()
    for k, v in kv.items():
        h[k.replace("_", "-")] = v
    return h


class _Ok:
    def __init__(self, status=200, **headers):
        self.status, self.headers = status, _headers(**headers)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b""


def _http_error(url, code, **headers):
    return urllib.error.HTTPError(url, code, "", _headers(**headers), io.BytesIO(b""))


class FakeGCS:
    """Enough of resumable GCS to tell a right client from a wrong one."""

    def __init__(self, *, fail_after: int | None = None):
        self.starts = 0
        self.sessions: dict[str, bytearray] = {}
        self.dead: set[str] = set()
        self.queries: list[str] = []
        self.fail_after = fail_after  # drop the connection once this many bytes landed

    def __call__(self, req: urllib.request.Request, timeout=None):
        url, method = req.full_url, req.get_method()
        if url.startswith(SIGNED):
            if method == "POST" and req.get_header("X-goog-resumable") == "start":
                self.starts += 1
                session = f"https://storage.example/session/{self.starts}"
                self.sessions[session] = bytearray()
                return _Ok(201, Location=session)
            # The production failure: a PUT against the signed start URL.
            raise _http_error(url, 400)
        if url not in self.sessions or url in self.dead:
            raise _http_error(url, 404)
        got = self.sessions[url]
        rng = req.get_header("Content-range", "")
        if method == "PUT" and rng.startswith("bytes */"):
            self.queries.append(url)
            if (
                got
                and rng.endswith(f"/{len(got)}")
                or (got and rng.endswith("/*") and self._done(url))
            ):
                return _Ok(200)
            if got:
                raise _http_error(url, 308, Range=f"bytes=0-{len(got) - 1}")
            raise _http_error(url, 308)
        start, _end, total = (
            int(x) for x in rng.removeprefix("bytes ").replace("-", "/").split("/")
        )
        assert start == len(got), f"chunk at {start} but {len(got)} committed"
        data = req.data
        if self.fail_after is not None and len(got) + len(data) > self.fail_after:
            got.extend(data[: self.fail_after - len(got)])
            self.fail_after = None
            raise urllib.error.URLError("connection reset")
        got.extend(data)
        self._totals = getattr(self, "_totals", {})
        self._totals[url] = total
        if len(got) == total:
            return _Ok(200)
        raise _http_error(url, 308, Range=f"bytes=0-{len(got) - 1}")

    def _done(self, url):
        return len(self.sessions[url]) == getattr(self, "_totals", {}).get(url, -1)


def _bundle(tmp_path, size=3 * 1024 * 1024 + 17) -> uploadmod.Bundle:
    path = tmp_path / "handoff" / "handoff.safetensors"
    path.parent.mkdir()
    path.write_bytes(bytes(i % 251 for i in range(size)))
    return uploadmod.Bundle(path, uploadmod.digest_file(path), size, workdir=tmp_path)


def _ticket() -> UploadTicket:
    return UploadTicket(signed_url=SIGNED, object_uri=OBJECT, chunk_bytes=1024 * 1024)


def test_progress_is_asked_of_the_session_never_the_signed_url(tmp_path, monkeypatch):
    gcs = FakeGCS()
    monkeypatch.setattr(urllib.request, "urlopen", gcs)
    bundle = _bundle(tmp_path)

    uploadmod.send(bundle, _ticket())

    assert gcs.starts == 1
    assert all(q.startswith("https://storage.example/session/") for q in gcs.queries)
    assert bytes(gcs.sessions["https://storage.example/session/1"]) == bundle.path.read_bytes()
    assert (tmp_path / uploadmod.SESSION_FILE).exists(), "keep completion until the sidecar lands"

    uploadmod.send(bundle, _ticket())
    assert gcs.starts == 1, "a retry recognizes the completed session instead of re-uploading"


def test_a_rerun_resumes_the_session_it_opened(tmp_path, monkeypatch):
    """A dropped connection at the last hop must not cost the bytes already sent."""
    gcs = FakeGCS(fail_after=1_500_000)
    monkeypatch.setattr(urllib.request, "urlopen", gcs)
    bundle = _bundle(tmp_path)

    with pytest.raises(AuditError, match="interrupted at byte"):
        uploadmod.send(bundle, _ticket())
    session_file = tmp_path / uploadmod.SESSION_FILE
    assert json.loads(session_file.read_text())["object"] == OBJECT
    committed = len(gcs.sessions["https://storage.example/session/1"])
    assert 0 < committed < bundle.size

    sent: list[int] = []
    uploadmod.send(bundle, _ticket(), on_progress=lambda done, total: sent.append(done))

    assert gcs.starts == 1, "resumed, not restarted"
    assert sent[0] > committed, "the first chunk of the re-run continues past what landed"
    assert bytes(gcs.sessions["https://storage.example/session/1"]) == bundle.path.read_bytes()
    assert session_file.exists(), "the sidecar has not landed yet"


def test_a_dead_session_is_replaced_and_a_foreign_one_ignored(tmp_path, monkeypatch):
    gcs = FakeGCS()
    monkeypatch.setattr(urllib.request, "urlopen", gcs)
    bundle = _bundle(tmp_path, 100)

    # A session saved by a run whose signed URL has since expired.
    (tmp_path / uploadmod.SESSION_FILE).write_text(
        json.dumps({"object": OBJECT, "session": "https://storage.example/session/expired"})
    )
    uploadmod.send(bundle, _ticket())
    assert gcs.starts == 1
    assert bytes(gcs.sessions["https://storage.example/session/1"]) == bundle.path.read_bytes()

    # A session for a different object (a new submission) is not reused either.
    (tmp_path / uploadmod.SESSION_FILE).write_text(
        json.dumps(
            {
                "object": "gs://intake/uploads/sub_other/handoff.safetensors",
                "digest": bundle.digest,
                "size": bundle.size,
                "session": "https://storage.example/session/1",
            }
        )
    )
    uploadmod.send(bundle, _ticket())
    assert gcs.starts == 2


def test_a_repacked_bundle_never_resumes_the_old_bundles_session(tmp_path, monkeypatch):
    """Resuming at the old offset with new bytes would upload a hybrid that no
    file on disk ever was, and the digest check would reject it after 24 GB."""
    gcs = FakeGCS(fail_after=1_500_000)
    monkeypatch.setattr(urllib.request, "urlopen", gcs)
    bundle = _bundle(tmp_path)
    with pytest.raises(AuditError):
        uploadmod.send(bundle, _ticket())
    saved = json.loads((tmp_path / uploadmod.SESSION_FILE).read_text())
    assert saved["digest"] == bundle.digest and saved["size"] == bundle.size

    # The auditor re-packs: same object, same size, different bytes.
    bundle.path.write_bytes(bytes((i + 7) % 251 for i in range(bundle.size)))
    repacked = uploadmod.Bundle(
        bundle.path, uploadmod.digest_file(bundle.path), bundle.size, workdir=tmp_path
    )
    assert repacked.digest != bundle.digest

    uploadmod.send(repacked, _ticket())
    assert gcs.starts == 2, "a fresh session for the new bytes"
    assert bytes(gcs.sessions["https://storage.example/session/2"]) == repacked.path.read_bytes()


def test_the_signed_url_itself_still_refuses_a_progress_query(tmp_path, monkeypatch):
    """Pin the failure mode so the fake cannot drift into forgiving it."""
    gcs = FakeGCS()
    monkeypatch.setattr(urllib.request, "urlopen", gcs)
    assert uploadmod._resume_offset(SIGNED) is None


# ── a spent token must not stop the hand-off ─────────────────────────────────


def _flow(tmp_path, monkeypatch, submit):
    """`_submit_and_upload` with the network stubbed: `submit` is what the record
    answers, everything after it is recorded rather than sent."""
    import json

    from gensyn_audit import cli, runner
    from gensyn_audit import plan as plan_mod
    from gensyn_audit import record as recordmod
    from gensyn_audit.outcome import Outcome

    bundle_file = tmp_path / "handoff.safetensors"
    bundle_file.write_bytes(b"tensors")
    calls = {"submit": [], "ticket": [], "send": [], "sidecar": []}
    monkeypatch.setattr(
        uploadmod,
        "build_bundle",
        lambda *a, **k: uploadmod.Bundle(bundle_file, "d" * 64, 7, tmp_path),
    )
    monkeypatch.setattr(uploadmod, "send", lambda *a, **k: calls["send"].append(a))
    monkeypatch.setattr(uploadmod, "send_sidecar", lambda p, t: calls["sidecar"].append(p))
    monkeypatch.setattr(uploadmod, "sidecar", lambda result, bundle, **_: {})
    monkeypatch.setattr(recordmod, "result_bundle", lambda **k: {"outcome": "match"})
    monkeypatch.setattr(cli, "_record_run_id", lambda *a: "run-id")

    class Rec:
        is_mock = False

        def submit(self, run, bundle, *, claim):
            calls["submit"].append(claim)
            return submit(claim)

        def upload_ticket(self, *a, **k):
            calls["ticket"].append(k.get("submission_id"))
            return UploadTicket(signed_url="https://x/b", sidecar_url="https://x/s")

    class Unit:
        is_init = False

    class Plan:
        unit = Unit()
        workdir = plan_mod.Workdir(tmp_path)

    state = runner.RunState(
        kit_id="k",
        kit_prefix="p",
        run="open-1b",
        unit_kind="interval",
        config_name="c",
        until_step=101,
        audit_step=100,
        expect_hash="a" * 64,
        device="cpu",
        pid=1,
        argv=[],
        env_overlay={},
        started_at="t",
        workdir=str(tmp_path),
        claim="clm_" + "1" * 64,
    )

    class Result:
        artifact = {"path": str(bundle_file)}

        def to_json(self):
            return json.dumps({})

    return cli, Plan(), state, Result(), Rec(), Outcome.MATCH, calls


@pytest.mark.parametrize("failure", [None, "build_bundle", "send", "send_sidecar"])
def test_completion_is_shown_only_after_both_uploads_succeed(
    tmp_path, monkeypatch, capsys, failure
):
    from gensyn_audit import record as recordmod

    response = recordmod.SubmitResponse(
        "pending-verification", "sub_1", "https://example.com/receipt", "", upload_required=True
    )
    cli, plan, state, result, rec, outcome, calls = _flow(
        tmp_path, monkeypatch, lambda claim: response
    )

    def fail(*args, **kwargs):
        raise AuditError("upload unavailable")

    next_command = "  gensyn-audit run --step 101"
    if failure:
        monkeypatch.setattr(uploadmod, failure, fail)
    if failure in ("send", "send_sidecar"):
        with pytest.raises(AuditError, match="upload unavailable"):
            cli._submit_and_upload(plan, state, result, rec, outcome)
    else:
        cli._submit_and_upload(plan, state, result, rec, outcome, next_command=next_command)
    out = capsys.readouterr().out
    assert ("AUDIT MATCHED. CONTRIBUTION SUBMITTED." in out) == (failure is None)
    assert (next_command in out) == (failure is None)
    if failure is None:
        assert calls["send"] and calls["sidecar"]
        assert out.index("AUDIT MATCHED.") < out.index("claim step 101 in the web app")
        assert "reused from this machine" in out
        assert "Checkpoint and sidecar uploaded" in out
        assert "Submitted for server verification." in out
        assert "Check the receipt for the latest status." in out
        assert out.rindex(response.receipt_url) < out.index(next_command)
        assert "\x1b" not in out and "█" not in out


@pytest.mark.parametrize("case", ["no-claim", "no-upload", "no-match", "superseded", "mock"])
def test_completion_does_not_overclaim(tmp_path, monkeypatch, capsys, case):
    from gensyn_audit import record as recordmod
    from gensyn_audit.outcome import Outcome

    response = recordmod.SubmitResponse(
        "superseded-pending-verification" if case == "superseded" else "pending-verification",
        "sub_1",
        None,
        "",
        upload_required=case != "no-upload",
        verify_mode="mock" if case == "mock" else None,
    )
    cli, plan, state, result, rec, outcome, _ = _flow(tmp_path, monkeypatch, lambda claim: response)
    if case == "no-claim":
        state.claim = None
    if case == "no-match":
        outcome = Outcome.NO_MATCH
    cli._submit_and_upload(plan, state, result, rec, outcome)
    assert "AUDIT MATCHED. CONTRIBUTION SUBMITTED." not in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["snapshot", "follow", "follow-upload", "follow-replay"])
@pytest.mark.parametrize("no_color", [False, True])
def test_status_renders_detached_completion_for_the_watching_terminal(
    tmp_path, monkeypatch, capsys, mode, no_color
):
    import time

    from gensyn_audit import brand, ui
    from gensyn_audit import record as recordmod

    monkeypatch.setattr(ui, "_FORCE_PLAIN", True)
    response = recordmod.SubmitResponse(
        "pending-verification", "sub_1", None, "", upload_required=True
    )
    cli, plan, state, result, rec, outcome, _ = _flow(tmp_path, monkeypatch, lambda claim: response)
    state.detached, state.pid, state.supervisor_pid = True, 0, 123
    state.finished_at = "2026-01-01T00:01:00+00:00"
    cli._submit_and_upload(plan, state, result, rec, outcome, next_command="next command")
    report = "result ---\n" + capsys.readouterr().out
    assert "\x1b" not in report and "█" not in report
    wd = plan.workdir
    saved_state = wd.state.read_bytes()
    matched = "state_hash=aaaa expected=aaaa MATCH=True\n"
    wd.log.write_text("" if mode == "follow-replay" else matched)
    active = mode in ("follow-upload", "follow-replay")
    wd.cli_log.write_text("result ---\npacking\n" if active else report)

    def advance(seconds):
        nonlocal active
        if not wd.log.read_text():
            wd.log.write_text(matched)
        else:
            wd.cli_log.write_text(report)
            active = False

    monkeypatch.setattr(cli.runner, "is_running", lambda pid: False)
    monkeypatch.setattr(cli.runner, "supervisor_running", lambda state: active)
    monkeypatch.setattr(time, "sleep", advance)
    monkeypatch.setattr(ui, "_FORCE_PLAIN", False)
    monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(brand, "width", lambda: 80)
    args = ["status", "--workdir", str(wd.root)]
    if mode != "snapshot":
        args.append("--follow")
    if no_color:
        args.append("--no-color")
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("AUDIT MATCHED. CONTRIBUTION SUBMITTED.") == 1
    assert out.index("AUDIT MATCHED.") < out.index("next command")
    assert "Submitted for server verification." in out
    assert "Check the receipt for the latest status." in out
    assert ("\x1b[38;2;250;215;209m" in out) == (not no_color)
    assert ("█" in out) == (not no_color)
    assert wd.cli_log.read_text() == report
    assert wd.state.read_bytes() == saved_state


@pytest.mark.parametrize("case", ["missing", "partial", "failed", "retry-failed"])
def test_status_never_infers_completion_from_a_match_or_an_earlier_attempt(
    tmp_path, monkeypatch, capsys, case
):
    from gensyn_audit import brand, cli, ui
    from gensyn_audit.plan import Workdir

    wd = Workdir(tmp_path)
    plain = "\n".join(brand.completion_banner(plain=True))
    report = "result ---\nMATCH\n"
    if case == "partial":
        report += plain.split("Check the receipt for the latest status.")[0]
    elif case == "retry-failed":
        report += plain + "\n===== supervisor started later =====\nprovisioning failed\n"
    elif case == "failed":
        report += "recorded — pending verification\nsidecar upload failed\n"
    if case != "missing":
        wd.cli_log.write_text(report)
    monkeypatch.setattr(ui, "_FORCE_PLAIN", False)
    monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(brand, "width", lambda: 80)
    cli._echo_supervisor_report(wd, completion_only=True)
    assert capsys.readouterr().out == ""
    cli._echo_supervisor_report(wd)
    assert "\x1b[38;2;250;215;209m" not in capsys.readouterr().out


def test_a_landed_submit_is_remembered_and_not_repeated(tmp_path, monkeypatch):
    """The token is spent by the first submit. A re-run to finish the hand-off
    must go straight to the upload with the id the record gave, or the
    documented "re-run the same command" path is a 401 dead end."""
    from gensyn_audit import record as recordmod
    from gensyn_audit import runner

    def submit(claim):
        if len(calls["submit"]) > 1:
            raise AuditError("POST results was refused (401 claim_not_active).")
        return recordmod.SubmitResponse(
            "pending-verification", "sub_1", None, "", upload_required=True
        )

    cli, plan, state, result, rec, outcome, calls = _flow(tmp_path, monkeypatch, submit)
    cli._submit_and_upload(plan, state, result, rec, outcome)
    assert calls["submit"] == [state.claim] and calls["ticket"] == ["sub_1"]
    saved = runner.load_state(plan.workdir)
    assert (saved.submission_id, saved.submitted_with) == ("sub_1", state.claim)

    # The connection dropped after that; the auditor re-runs.
    cli._submit_and_upload(plan, saved, result, rec, outcome)
    assert calls["submit"] == [state.claim], "no second POST with a spent token"
    assert calls["ticket"] == ["sub_1", "sub_1"] and len(calls["send"]) == 2


def test_session_is_forgotten_only_after_the_sidecar_lands(tmp_path, monkeypatch):
    from gensyn_audit import record as recordmod
    from gensyn_audit import runner

    def submit(claim):
        return recordmod.SubmitResponse(
            "pending-verification", "sub_1", None, "", upload_required=True
        )

    cli, plan, state, result, rec, outcome, calls = _flow(tmp_path, monkeypatch, submit)
    forgotten = []
    monkeypatch.setattr(uploadmod, "_forget_session", lambda bundle: forgotten.append(bundle))

    def send_sidecar(*args):
        calls["sidecar"].append(args)
        if len(calls["sidecar"]) == 1:
            raise AuditError("sidecar interrupted")

    monkeypatch.setattr(uploadmod, "send_sidecar", send_sidecar)
    with pytest.raises(AuditError, match="sidecar interrupted"):
        cli._submit_and_upload(plan, state, result, rec, outcome)
    assert forgotten == []

    cli._submit_and_upload(plan, runner.load_state(plan.workdir), result, rec, outcome)
    assert len(forgotten) == 1


def test_a_new_claim_submits_again_rather_than_reusing_an_old_submission(tmp_path, monkeypatch):
    from gensyn_audit import record as recordmod

    def submit(claim):
        return recordmod.SubmitResponse(
            "pending-verification", f"sub_{len(calls['submit'])}", None, "", upload_required=True
        )

    cli, plan, state, result, rec, outcome, calls = _flow(tmp_path, monkeypatch, submit)
    state.submission_id, state.submitted_with = "sub_old", "clm_" + "0" * 64
    cli._submit_and_upload(plan, state, result, rec, outcome)
    assert calls["submit"] == [state.claim] and calls["ticket"] == ["sub_1"]


def test_upload_accepts_the_submission_id_the_record_returned():
    """For a result an older build submitted without remembering the id."""
    from gensyn_audit import cli

    args = cli.build_parser().parse_args(
        ["upload", "--claim", "clm_x", "--submission", "sub_0132a5035ac5b04d06cb706a"]
    )
    assert args.submission == "sub_0132a5035ac5b04d06cb706a"


def test_a_result_the_record_did_not_want_uploaded_is_not_remembered(tmp_path, monkeypatch):
    """`recorded-no-public-change` with no upload requested must not become a
    pending upload on the next run: the record declined the hand-off, and the
    CLI would otherwise push a bundle it never asked for."""
    from gensyn_audit import record as recordmod

    def submit(claim):
        return recordmod.SubmitResponse(
            "recorded-no-public-change", "sub_declined", None, "", upload_required=False
        )

    cli, plan, state, result, rec, outcome, calls = _flow(tmp_path, monkeypatch, submit)
    cli._submit_and_upload(plan, state, result, rec, outcome)
    assert calls["ticket"] == [] and calls["send"] == []
    assert state.submission_id is None
    assert not plan.workdir.state.is_file(), "nothing to remember, so nothing written"

    # The re-run submits again (and is refused or recorded again by the record),
    # rather than fabricating an upload for the declined result.
    cli._submit_and_upload(plan, state, result, rec, outcome)
    assert calls["submit"] == [state.claim, state.claim]
    assert calls["ticket"] == [] and calls["send"] == []


def test_the_sidecar_names_the_run_by_id_even_when_addressed_by_name(tmp_path, monkeypatch):
    """The receipt's `run` is `open-1b` when the manifest addressed the record
    by name; the verifier's RUN_ID is the id. The sidecar must carry the id."""
    from gensyn_audit import cli

    seen = {}
    monkeypatch.setattr(
        uploadmod,
        "build_bundle",
        lambda *a, **k: uploadmod.Bundle(tmp_path / "b", "d" * 64, 1, tmp_path),
    )
    monkeypatch.setattr(uploadmod, "send", lambda *a, **k: None)
    monkeypatch.setattr(uploadmod, "send_sidecar", lambda payload, t: seen.update(payload))

    class Manifest:
        run_id = "20260722-213626-ad3276b"

    class Rec:
        is_mock = False

        def manifest(self):
            return Manifest()

        def upload_ticket(self, *a, **k):
            return UploadTicket(signed_url="https://x/b", sidecar_url="https://x/s")

    class Plan:
        class workdir:
            handoff = tmp_path
            root = tmp_path

    class State:
        run, until_step, claim = "open-1b", 101, "clm_x"

    class Result:
        artifact = {"path": str(tmp_path / "b")}
        run, audit_step, reproduced_hash, committed_hash = "open-1b", 100, "a" * 64, "a" * 64
        matched, losses, repop_commit, repop_backends = True, [], "abc", ["cuda"]
        device, tool = "cuda", "gensyn-audit/0.1.0"

    (tmp_path / "b").write_bytes(b"x")
    cli._do_upload(Plan(), State(), Rec(), None, Result())
    assert seen["run_id"] == "20260722-213626-ad3276b"
    assert seen["step"] == 100


def test_a_manifest_failure_stops_before_building_the_bundle(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gensyn_audit import cli

    def fail():
        raise AuditError("manifest unavailable")

    built = []
    monkeypatch.setattr(uploadmod, "build_bundle", lambda *a, **k: built.append(True))
    record = SimpleNamespace(is_mock=False, manifest=fail)
    plan = SimpleNamespace(workdir=SimpleNamespace(root=tmp_path))
    with pytest.raises(AuditError, match="manifest unavailable"):
        cli._do_upload(plan, None, record, None, SimpleNamespace(run="open-1b"))
    assert built == []
