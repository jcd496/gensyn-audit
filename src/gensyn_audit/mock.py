"""A stand-in record, for the API that does not exist yet.

`audit.gensyn.ai` and `open1b.gensyn.ai` do not resolve and no endpoint answers
the PRD's contract, so this implements the same `Record` protocol in-process
against a JSON file. It exists so the rest of the tool can be built and tested
against a real seam rather than a stub, and so switching to the live API is a
`--record <url>` away.

Two rules govern everything here.

**It must never pass for the real thing.** Every surface says MOCK: the label,
a banner on each call, and a `mocked` block written into the receipt. A tool
that quietly fakes a submission is worse than one that cannot submit.

**Its gate must be real.** The record's whole verification is comparing the
replay's losses against values it withholds. A mock that accepted anything
would let the CLI's most important path go untested, so this one holds an
actual withheld series (extracted from a real run's `metrics.jsonl`) and
compares for exact equality, exactly as the backend is specified to.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import record as recordmod
from .errors import AuditError
from .outcome import NO_MATCH_EDUCATION as NO_MATCH_GUIDANCE
from .record import _ERROR_GUIDANCE as _ERROR_HINTS
from .record import Predecessor, StepContext, SubmitResponse, UploadTicket
from .ui import paint

BANNER = "MOCK RECORD — nothing here reaches a real service"


@dataclass
class MockState:
    """What the fake backend remembers between invocations."""

    path: Path
    accepted: dict = field(default_factory=dict)
    """step -> the submission that landed first, for `superseded`."""

    used_claims: list = field(default_factory=list)
    """Claim tokens already spent. Submission is single-use."""

    uploads: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> MockState:
        if not path.is_file():
            return cls(path=path)
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls(path=path)
        return cls(
            path=path,
            accepted=doc.get("accepted", {}),
            used_claims=doc.get("used_claims", []),
            uploads=doc.get("uploads", {}),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "_comment": "State of the MOCK record. Not a real audit ledger.",
                    "accepted": self.accepted,
                    "used_claims": self.used_claims,
                    "uploads": self.uploads,
                },
                indent=2,
            )
            + "\n"
        )


class MockRecord:
    """The `Record` protocol, backed by local files.

    `fixture` is a JSON document holding the step contexts this mock can serve
    and the withheld loss series it gates on -- the two things a real backend
    would hold in a database and a private bucket respectively.
    """

    is_mock = True

    def __init__(self, fixture: Path, state_path: Path | None = None) -> None:
        if not fixture.is_file():
            raise AuditError(
                f"no mock fixture at {fixture}.",
                hint="The mock record needs a fixture describing the steps it can serve.",
            )
        try:
            self.doc = json.loads(fixture.read_text())
        except json.JSONDecodeError as exc:
            raise AuditError(f"mock fixture {fixture} is not valid JSON: {exc}") from exc
        self.fixture = fixture
        self.label = f"mock://{fixture}"
        self.state = MockState.load(state_path or fixture.parent / "mock-record-state.json")
        self.intake = fixture.parent / "mock-intake"

    # ── announcement ─────────────────────────────────────────────────────────
    @staticmethod
    def banner() -> str:
        return paint(f"  ⚠ {BANNER}", "yellow")

    def provenance(self) -> dict:
        """Recorded in the receipt so a reader can never mistake this for real."""
        return {
            "record": "mock",
            "fixture": str(self.fixture),
            "note": "No real service was contacted. Statuses and receipt URLs below "
            "are produced locally by gensyn_audit.mock.",
        }

    # ── the protocol ─────────────────────────────────────────────────────────
    def step_context(self, run: str, step: int) -> StepContext:
        steps = self.doc.get("steps", {})
        raw = steps.get(str(step))
        if raw is None:
            known = ", ".join(sorted(steps, key=int)) or "none"
            raise AuditError(
                f"the mock record has no context for step {step}.",
                hint=f"Fixture {self.fixture.name} covers: {known}",
            )
        pred = raw["predecessor"]
        return StepContext(
            run=run,
            step=step,
            committed_hash=str(raw.get("committed") or raw["committed_hash"]).lower(),
            predecessor=Predecessor(
                source=pred.get("kind") or pred.get("source") or "",
                uri=pred.get("uri"),
                digest=(str(pred["digest"]).lower() if pred.get("digest") else None),
                step=int(pred["step"]) if pred.get("step") is not None else None,
                check=pred.get("check"),
                artifact_files=recordmod.parse_artifact_files(pred.get("artifactFiles")),
            ),
            descriptor_uri=raw.get("descriptorCheckpointUri") or raw.get("descriptor_uri"),
            phase=raw.get("phase"),
            microbatches=raw.get("microbatches"),
            gcs_root=raw.get("gcs_root") or self.doc.get("gcs_root"),
            state=raw.get("state", "unaudited"),
            is_frontier=bool(raw.get("isFrontier", True)),
            closes_segment=bool(raw.get("closesSegment", False)),
            corroboration_only=bool(raw.get("corroborationOnly", False)),
            expected_runtime_hours=raw.get("expectedRuntimeHours"),
        )

    def segment_fork(self, run: str, segment: int):
        from .record import _parse_fork

        del run
        return _parse_fork((self.doc.get("segments") or {}).get(str(segment), {}).get("fork"))

    def submit(self, run: str, bundle: dict, *, claim: str) -> SubmitResponse:
        step = str(bundle.get("step"))

        if claim in self.state.used_claims:
            # The real API is NOT idempotent here: the token is single-use, so a
            # retry after a lost response gets 401 claim_not_active and the
            # auditor is told to read the receipt instead. Mirroring that
            # matters more than being forgiving — a mock that quietly accepted
            # a replay would hide the one case the CLI has to handle well.
            raise AuditError(
                "POST results was refused (401 claim_not_active).",
                hint=_ERROR_HINTS["claim_not_active"],
            )

        outcome = bundle.get("outcome")
        if outcome != "match":
            # Recorded for triage; nothing public changes and the claim is
            # released, so the step returns to the pool.
            return SubmitResponse(
                disposition="recorded-no-public-change",
                submission_id=f"sub_{_short(claim)}",
                receipt_url=f"mock://receipt/{run}/{step}",
                detail=f"outcome {outcome!r} does not advance the record.",
                guidance=NO_MATCH_GUIDANCE,
            )

        if bundle.get("reportedStateHash") != (
            self.doc.get("steps", {}).get(step, {}).get("committed")
            or self.doc.get("steps", {}).get(step, {}).get("committed_hash")
        ):
            raise AuditError(
                "POST results was refused (422 hash_contradicts_outcome).",
                hint=_ERROR_HINTS["hash_contradicts_outcome"],
            )

        if problem := self._check_losses(step, bundle):
            return SubmitResponse(
                disposition="recorded-no-public-change",
                submission_id=f"sub_{_short(claim)}",
                receipt_url=None,
                detail=problem,
            )

        sub = f"sub_{_short(claim)}"
        superseded = step in self.state.accepted
        receipt = f"mock://receipt/{run}/{step}/{_short(claim)}"
        self.state.accepted.setdefault(
            step, {"claim": claim, "receipt_url": receipt, "at": int(time.time())}
        )
        self.state.used_claims.append(claim)
        self.state.uploads.setdefault(sub, {"step": step, "digest": bundle.get("artifactDigest")})
        self.state.save()
        return SubmitResponse(
            disposition=(
                "superseded-pending-verification" if superseded else "pending-verification"
            ),
            submission_id=sub,
            receipt_url=receipt,
            detail=(
                "another auditor's match is already accepted for this step."
                if superseded
                else "recorded; the loss gate runs asynchronously."
            ),
            upload_required=bool(bundle.get("artifactDigest")),
            upload_endpoint="/v1/uploads",
            verify_mode="mock",
            public_state_changed=False,
        )

    def _check_losses(self, step: str, bundle: dict) -> str | None:
        """The real gate, against a withheld series. None means it passed."""
        withheld = (self.doc.get("withheld_losses") or {}).get(step)
        if withheld is None:
            return (
                f"no withheld losses for step {step} in this fixture, so a match cannot be gated."
            )
        if bundle.get("ce") is None or bundle.get("zLoss") is None:
            return (
                "a match must carry ce and zLoss: the committed hash is "
                "public, so echoing it is not evidence of work."
            )
        for sent, key in (("ce", "loss_ce"), ("zLoss", "loss_zloss")):
            if bundle[sent] != withheld[key]:
                return (
                    f"{sent} does not match the cluster's value for step "
                    f"{step}. Under bitwise reproducibility these are "
                    "identical, so this replay did not reproduce the step."
                )
        return None

    def upload_ticket(
        self,
        run: str,
        step: int,
        *,
        claim: str,
        size: int,
        digest: str,
        submission_id: str | None = None,
    ) -> UploadTicket:
        del run, size, submission_id
        if claim not in self.state.used_claims:
            raise AuditError("an upload ticket needs a recorded submission first.")
        self.intake.mkdir(parents=True, exist_ok=True)
        target = self.intake / f"step{step}-{digest[:12]}-handoff.safetensors"
        sidecar = target.with_name(target.name.replace("handoff.safetensors", "handoff.json"))
        self.state.uploads[f"sub_{_short(claim)}"] = {
            "step": str(step),
            "target": str(target),
            "digest": digest,
            "sidecar": str(sidecar),
        }
        self.state.save()
        # file:// so upload.py's resumable path is exercised for real rather
        # than short-circuited.
        return UploadTicket(
            signed_url=f"file://{target}",
            object_uri=f"gs://mock-intake/uploads/step{step}",
            sidecar_url=f"file://{sidecar}",
            sidecar_uri=f"gs://mock-intake/uploads/step{step}/handoff.json",
        )

    def verify_sidecar(self, step: int) -> str | None:
        """The verifier's presence and consistency check on ``handoff.json``:
        it must exist, and its ``bundle_digest`` must be the declared one."""
        rec = next(
            (
                v
                for v in self.state.uploads.values()
                if v.get("step") == str(step) and v.get("sidecar")
            ),
            None,
        )
        if not rec or not Path(rec["sidecar"]).is_file():
            return "handoff.json did not arrive; the verifier rejects with missing_artifact."
        declared = json.loads(Path(rec["sidecar"]).read_text())
        if declared.get("bundle_digest") != rec["digest"]:
            return "handoff.json declares a different bundle_digest than the submission."
        return None

    def verify_upload(self, step: int) -> str | None:
        """What the backend does once the bytes land: check them against the
        digest declared at submit time, before publishing as the next
        predecessor. blake2b-256, per the contract."""
        rec = next(
            (
                v
                for v in self.state.uploads.values()
                if v.get("step") == str(step) and v.get("target")
            ),
            None,
        )
        if not rec:
            return "no upload recorded."
        target = Path(rec["target"])
        if not target.is_file():
            return "the upload did not arrive."
        got = hashlib.blake2b(target.read_bytes(), digest_size=32).hexdigest()
        if got != rec["digest"]:
            return f"uploaded bytes hash to {got[:16]}, declared {rec['digest'][:16]}."
        return None


def _short(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:10]


def default_fixture() -> Path | None:
    env = os.environ.get("AUDIT_MOCK_FIXTURE")
    if env:
        return Path(env).expanduser()
    # Present in a source checkout, absent from a built wheel on purpose: the
    # fixture carries the real withheld loss series and must not be
    # distributed.
    bundled = Path(__file__).parent / "mocks" / "open-1b.json"
    return bundled if bundled.is_file() else None
