"""Client and data types for the audit record service.

The web app mints claim tokens; the CLI uses one to submit a finished replay
and, when requested by the service, obtain an upload ticket for its handoff.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from . import __version__
from .commitments import Proof, ProofLink
from .errors import AuditError
from .kit import SHA256, KitFile
from .outcome import Outcome


@dataclass(frozen=True)
class RecordManifest:
    """What the record says about itself, from ``GET /manifest.json``.

    ``simulated`` identifies records that use placeholder commitments or gates.
    Replaying against a placeholder commitment cannot produce a meaningful
    match, so the CLI surfaces it before starting.
    """

    run_id: str
    run_name: str
    total_steps: int | None
    steps_per_segment: int | None
    artifact_check: str | None
    claim_ttl_hours: int | None
    exchange_format: str | None
    simulated: bool
    simulated_parts: dict
    endpoints: dict

    @property
    def commitments_are_placeholders(self) -> bool:
        # `simulatedParts` carries every part as a key: a string explains what is
        # simulated, `null` means that part is real. Testing for the key alone
        # reported real commitments as placeholders -- the loudest warning this
        # tool has, fired against a genuine hash.
        return self.simulated_parts.get("commitments") is not None

    @property
    def loss_gate_is_live(self) -> bool:
        return self.simulated_parts.get("lossGate") is None


@dataclass(frozen=True)
class Fork:
    """A mid-run change to something segment-scoped, and the descriptor for it.

    A fork splits a run: the clipper changed, or `repop_env` did, or the
    topology. One replay carries exactly one descriptor, so the single interval
    that crosses the boundary must load its STATE from the old segment's
    checkpoint but its DESCRIPTOR from the new segment's. Miss that and the
    replay runs the interval under the old rules and mismatches on the first
    step -- a "divergence" that is really a configuration error.

    It lives on the segment, not the step receipt, so it takes a second call.
    """

    boundary_step: int
    descriptor_checkpoint_uri: str
    note: str | None = None

    def applies_to(self, predecessor_step: int) -> bool:
        """Exactly one interval per fork needs the overlay: the one starting at
        the boundary. Everything before it is ordinary under the old descriptor,
        everything after records its own."""
        return predecessor_step == self.boundary_step


#: How a predecessor is trusted, decided by the record and by nothing else.
#: These are not cosmetic labels: an ANCHOR skips the tensor-state gate, so
#: anything that is not unambiguously one must not be treated as one.
ANCHOR, CROWD, UNKNOWN = "gensyn-anchor", "crowd", "unknown"

#: The record's vocabulary for `predecessor.kind`, mapped to the two paths.
#: Matching is exact and closed — a value the record does not publish today is
#: UNKNOWN, never "probably an anchor". A hostname, an absent gradients file
#: or a flag inside the artifact are never inputs to this.
_KINDS = {
    "published": ANCHOR,
    "published checkpoint": ANCHOR,
    "gensyn-anchor": ANCHOR,
    "crowd": CROWD,
    "crowd-provided": CROWD,
    "crowd-provided checkpoint": CROWD,
}


@dataclass(frozen=True)
class Predecessor:
    """Where this step starts from, and what its bytes must hash to.

    ``source`` is the whole reason step context cannot be a static file: for a
    step inside an open segment this is another auditor's uploaded checkpoint,
    which did not exist when the run finished.
    """

    source: str
    """``published checkpoint`` (a run anchor) or ``crowd-provided checkpoint``."""

    uri: str | None
    """Null while the previous auditor's hand-off does not exist yet."""

    digest: str | None
    """BLAKE2b-256 of a packed crowd hand-off, declared as artifactDigest by
    its uploader and published by the record after acceptance. Not the
    training-state commitment. Published anchors currently carry null."""

    step: int | None
    check: str | None = None

    artifact_files: tuple[KitFile, ...] = ()
    """Optional per-file SHA-256 manifest for directory checkpoints.
    Packed crowd hand-offs use ``digest`` and do not require this extension."""

    @property
    def provenance(self) -> str:
        return _KINDS.get((self.source or "").strip().lower(), UNKNOWN)

    @property
    def is_crowd_provided(self) -> bool:
        return self.provenance == CROWD

    @property
    def has_bundle_digest(self) -> bool:
        return (
            isinstance(self.digest, str) and re.fullmatch(r"[0-9a-f]{64}", self.digest) is not None
        )

    @property
    def is_trusted_anchor(self) -> bool:
        """An original Gensyn checkpoint, as identified by the record itself."""
        return self.provenance == ANCHOR

    @property
    def exists(self) -> bool:
        return bool(self.uri)


@dataclass(frozen=True)
class StepContext:
    """One step's mutable coordination state, from the API.

    The committed hash is also in the run's static ``state_hashes.jsonl``, which
    stays authoritative; it is returned here so a runner needs one call rather
    than two. Do not treat the API as the source of truth for a commitment.
    """

    run: str
    step: int
    committed_hash: str
    predecessor: Predecessor
    descriptor_uri: str | None
    phase: str | None
    microbatches: int | None
    gcs_root: str | None
    segment: int | None = None
    state: str | None = None
    """One of the record's four public states: unaudited, claimed, provisional,
    confirmed. Load-bearing vocabulary — never render it as "verified"."""

    is_frontier: bool = True
    closes_segment: bool = False
    corroboration_only: bool = False
    """The last step of a segment reproduces an already-published checkpoint, so
    auditing it corroborates rather than proves. Say so."""

    expected_runtime_hours: float | None = None

    proof: Proof | None = None
    """Inclusion of ``committed_hash`` in the segment's Merkle root."""


@dataclass(frozen=True)
class SubmitResponse:
    """What the record says when it takes a result.

    Nothing is accepted synchronously. The losses are checked by a separate
    verification service, so a successful submit means *recorded, pending
    verification* -- and calling that "accepted" would tell the auditor their
    audit landed when it has not been checked yet.
    """

    disposition: str
    """``pending-verification`` | ``superseded-pending-verification`` |
    ``recorded-no-public-change``."""

    submission_id: str | None
    receipt_url: str | None
    detail: str
    guidance: str = ""
    upload_required: bool = False
    upload_endpoint: str | None = None
    verify_mode: str | None = None
    """Verification mode reported by the record service."""

    public_state_changed: bool = False

    @property
    def recorded(self) -> bool:
        """It reached the record. Not the same as passing the loss gate."""
        return self.disposition.startswith(("pending", "superseded"))

    @property
    def superseded(self) -> bool:
        """Another auditor's match is already accepted. Done and corroborating,
        never a failure."""
        return self.disposition.startswith("superseded")


@dataclass(frozen=True)
class UploadTicket:
    signed_url: str
    """A GCS V4 signed URL, valid 1h, that *starts* a resumable session: POST it
    with ``x-goog-resumable: start``, then PUT the bundle to the session URI
    that comes back in ``Location``. Bytes never pass through the API."""

    object_uri: str | None = None
    chunk_bytes: int = 8 * 1024 * 1024

    sidecar_url: str | None = None
    """A plain signed PUT for ``handoff.json`` beside the bundle. The verifier
    requires both objects and rejects a submission with one as
    ``missing_artifact``, so a record that offers no sidecar URL cannot accept
    an upload from this tool."""

    sidecar_uri: str | None = None


class Record(Protocol):
    """What the runner needs from the record. Implemented twice."""

    label: str
    """Shown to the auditor, so a mocked backend can never pass for a real one."""

    is_mock: bool

    def step_context(self, run: str, step: int) -> StepContext: ...

    def segment_fork(self, run: str, segment: int) -> Fork | None: ...

    def submit(self, run: str, bundle: dict, *, claim: str) -> SubmitResponse: ...

    def upload_ticket(
        self,
        run: str,
        step: int,
        *,
        claim: str,
        size: int,
        digest: str,
        submission_id: str | None = None,
    ) -> UploadTicket: ...


# ── the real client ──────────────────────────────────────────────────────────


class HttpRecord:
    """The audit record's HTTP client."""

    is_mock = False

    def __init__(self, base_url: str, *, timeout: int = 60) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.label = self.base

    def _request(
        self, method: str, path: str, *, body: dict | None = None, claim: str | None = None
    ) -> dict:
        raw = self._request_text(method, path, body=body, claim=claim)
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            if "error code: 1010" in raw:
                raise AuditError(
                    "Cloudflare rejected the request before it reached the API.",
                    hint="That is the Browser Integrity Check refusing a generic "
                    "user agent. This client sends its own; a proxy may be "
                    "rewriting it.",
                ) from exc
            raise AuditError(f"{path} returned non-JSON: {raw[:200]!r}") from exc

    def _request_text(
        self, method: str, path: str, *, body: dict | None = None, claim: str | None = None
    ) -> str:
        url = f"{self.base}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        # Both hosts sit behind Cloudflare's Browser Integrity Check, which
        # answers a stock `Python-urllib/3.x` agent with 403 and `error code:
        # 1010` before the request reaches the API at all. Identify the tool.
        req.add_header("User-Agent", f"gensyn-audit/{__version__}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if claim:
            req.add_header("Authorization", f"Bearer {claim}")
        for key, value in _extra_headers().items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise _api_error(exc, method, path) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AuditError(
                f"could not reach the record at {self.base}: {exc}",
                hint="The replay is finished and the receipt is on disk — nothing "
                "is lost. The claim token is single-use, so check the receipt "
                "before re-submitting.",
            ) from exc
        return raw

    def ledger(self, *, after: int | None = None, limit: int = 50) -> list[dict]:
        """The machine-readable ledger. JSONL, so one row per line.

        Not JSON: `json.loads` over the whole body fails on the second line,
        which is how this was broken until it was actually called.
        """
        query = {"limit": str(limit)}
        if after is not None:
            query["after"] = str(after)
        raw = self._request_text("GET", f"/v1/ledger.jsonl?{urllib.parse.urlencode(query)}")
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a truncated tail is not worth failing over
        return rows

    def manifest(self) -> RecordManifest:
        return _parse_manifest(self._request("GET", "/manifest.json"))

    def step_context(self, run: str, step: int) -> StepContext:
        doc = self._request("GET", f"/v1/runs/{run}/steps/{step}/receipt")
        return _parse_step_context(run, step, doc)

    def segment_fork(self, run: str, segment: int) -> Fork | None:
        """The fork descriptor for a segment, if it straddles a boundary."""
        doc = self._request("GET", f"/v1/runs/{run}/segments/{segment}")
        return _parse_fork(doc.get("fork"))

    def submit(self, run: str, bundle: dict, *, claim: str) -> SubmitResponse:
        doc = self._request(
            "POST", f"/v1/runs/{run}/steps/{bundle['step']}/results", body=bundle, claim=claim
        )
        upload = doc.get("upload") or {}
        verify = doc.get("verify") or {}
        return SubmitResponse(
            disposition=str(doc.get("disposition", "recorded-no-public-change")),
            submission_id=doc.get("submissionId"),
            receipt_url=doc.get("receiptUrl"),
            detail=str(doc.get("detail", "")),
            guidance=str(doc.get("guidance", "")),
            upload_required=bool(upload.get("required", False)),
            upload_endpoint=upload.get("endpoint"),
            verify_mode=verify.get("mode"),
            public_state_changed=bool(doc.get("publicStateChanged", False)),
        )

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
        # The digest was declared at submit time as `artifactDigest`; this call
        # only names the submission it belongs to.
        del run, step, size, digest
        if not submission_id:
            raise AuditError("an upload needs the submissionId the record returned at submit.")
        doc = self._request(
            "POST", "/v1/uploads", claim=claim, body={"submissionId": submission_id}
        )
        if not doc.get("signedUrl"):
            raise AuditError("the record returned no signed URL.", hint=json.dumps(doc)[:300])
        sidecar = doc.get("sidecar") or {}
        return UploadTicket(
            signed_url=doc["signedUrl"],
            object_uri=doc.get("objectUri"),
            sidecar_url=sidecar.get("signedUrl"),
            sidecar_uri=sidecar.get("objectUri"),
        )


#: Stable error codes the API returns, and what they mean for the auditor.
#: Branch on these, never on the human-readable `detail`.
_ERROR_GUIDANCE = {
    "no_token": "This call needs a claim token and none was sent. Claim the step "
    "in the web app and pass the token as --claim; a replay without "
    "one is still reported locally, but the record never hears about "
    "it.",
    "bad_request": "The record rejected the shape of this request, not your "
    "audit. The detail below says which field; if it names one "
    "this tool sends, the client and the API disagree about the "
    "contract and that is worth reporting rather than retrying.",
    "bad_token": "That claim token was not recognised. Copy it again from the web app's Run stage.",
    "claim_not_active": "This claim is spent or released. Submission is "
    "single-use: if a previous submit reached the record, "
    "your result is already there, awaiting verification. "
    "It does not show on the receipt until the hand-off is "
    "checked. To finish the hand-off, re-run in the same "
    "workdir, or `gensyn-audit upload --claim <token> "
    "--submission <sub_id>` with the id the record returned.",
    "claim_step_mismatch": "This token was minted for a different step.",
    "not_frontier": "This step is not the frontier: an earlier step in the "
    "segment is still unaudited.",
    "segment_closed": "That segment is already closed.",
    "hash_contradicts_outcome": "A `match` must carry the committed hash. If "
    "yours differs, submit `no-match` — that is a "
    "real outcome, not a failure to report.",
    "rate_limited": "Too many requests from this address. Wait and retry.",
    "no_digest_declared": "An upload needs `artifactDigest` declared at submit time.",
    "already_uploaded": "This submission already has an artifact.",
    "uploads_unconfigured": "The record cannot accept uploads yet (the intake "
    "bucket is not configured). Your result is already "
    "recorded; only the hand-off is missing.",
}


def _api_error(exc: urllib.error.HTTPError, method: str, path: str) -> AuditError:
    """Turn the API's `{error, detail}` into something actionable."""
    raw = exc.read().decode("utf-8", errors="replace")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        doc = {}
    code = str(doc.get("error") or "")
    detail = doc.get("detail") or raw[:300]
    hint = _ERROR_GUIDANCE.get(code, "")
    if code == "rate_limited" and doc.get("retry_after"):
        hint += f" Retry after {doc['retry_after']}s."
    return AuditError(
        f"{method} {path} was refused ({exc.code}" + (f" {code}" if code else "") + ").",
        hint=(hint + ("\n" + detail if detail and hint else detail)).strip() or None,
    )


def _extra_headers() -> dict[str, str]:
    """Extra headers from `AUDIT_HTTP_HEADERS` (``K: V`` per line).

    Staging sits behind Cloudflare Access, which a CLI cannot complete on its
    own. This is the seam for a `cf-access-token` obtained out of band, and it
    is deliberately generic rather than a CF-specific flag: production is not
    expected to need it.
    """
    raw = os.environ.get("AUDIT_HTTP_HEADERS", "")
    out = {}
    for line in raw.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            out[key.strip()] = value.strip()
    return out


def _parse_manifest(doc: dict) -> RecordManifest:
    run = doc.get("run") or {}
    audit = doc.get("audit") or {}
    return RecordManifest(
        run_id=str(run.get("id", "")),
        # The endpoint index addresses the run by NAME (`/v1/runs/open-1b/…`),
        # not by id. Both are accepted today; the name is what the record
        # publishes, so prefer it.
        run_name=str(run.get("name") or run.get("id", "")),
        total_steps=run.get("totalSteps"),
        steps_per_segment=run.get("stepsPerSegment"),
        artifact_check=audit.get("artifactCheck"),
        claim_ttl_hours=audit.get("claimTtlHours"),
        exchange_format=audit.get("exchangeFormat"),
        simulated=bool(doc.get("simulated", False)),
        simulated_parts=doc.get("simulatedParts") or {},
        endpoints=doc.get("endpoints") or {},
    )


def _parse_fork(raw: object) -> Fork | None:
    if not isinstance(raw, dict) or not raw.get("descriptorCheckpointUri"):
        return None
    return Fork(
        boundary_step=int(raw["boundaryStep"]),
        descriptor_checkpoint_uri=str(raw["descriptorCheckpointUri"]),
        note=raw.get("note"),
    )


def _parse_proof(raw: object) -> Proof | None:
    if not isinstance(raw, dict) or not raw.get("leaf") or not raw.get("root"):
        return None
    return Proof(
        leaf=str(raw["leaf"]).lower(),
        root=str(raw["root"]).lower(),
        path=tuple(
            ProofLink(hash=str(link["hash"]).lower(), side=str(link.get("side", "")))
            for link in (raw.get("path") or [])
            if isinstance(link, dict) and link.get("hash")
        ),
    )


def parse_artifact_files(raw: object) -> tuple[KitFile, ...]:
    """``predecessor.artifactFiles`` -> the digests to hold the download to.

    Same three fields ``kit.json`` publishes per wheel (``name``, ``sha256``,
    ``bytes``), because a checkpoint directory needs exactly what a kit needs:
    a digest per file, from a source other than the file. A malformed entry is
    dropped. The integrity gate requires every on-disk file to be listed, so
    dropping an entry cannot silently admit that file.
    """
    out = []
    if not isinstance(raw, (list, tuple)):
        return ()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name, sha = str(entry.get("name", "")), str(entry.get("sha256", "")).lower()
        if name and SHA256.match(sha):
            try:
                size = int(entry.get("bytes", 0))
            except (TypeError, ValueError, OverflowError):
                continue
            if size >= 0:
                out.append(KitFile(name=name, sha256=sha, bytes=size))
    return tuple(out)


def _parse_step_context(run: str, step: int, doc: dict) -> StepContext:
    pred = doc.get("predecessor") or {}
    return StepContext(
        run=run,
        step=int(doc.get("step", step)),
        committed_hash=str(doc.get("committed") or doc.get("committed_hash") or "").lower(),
        predecessor=Predecessor(
            # `uri` and `digest` are null while the previous auditor's hand-off
            # does not exist. That is a real state of the relay, not an error:
            # it means nobody has produced this step's starting point yet.
            # No default. A receipt that does not say what the predecessor is
            # is not saying "anchor": defaulting to the one path that skips
            # the state-hash gate would make silence the way past it.
            source=str(pred.get("kind") or pred.get("source") or ""),
            uri=(str(pred["uri"]) if pred.get("uri") else None),
            digest=(str(pred["digest"]).lower() if pred.get("digest") else None),
            # No default. `step - 1` looks harmless but silently mixes the
            # numberings: `step` here is the record's audit number, and
            # subtracting one from it yields neither the audit nor the log
            # number of the predecessor. Absent means absent; the caller
            # derives it from the step numbering, which knows the difference.
            step=(int(pred["step"]) if pred.get("step") is not None else None),
            check=pred.get("check"),
            artifact_files=parse_artifact_files(pred.get("artifactFiles")),
        ),
        descriptor_uri=doc.get("descriptorCheckpointUri") or doc.get("descriptor_uri"),
        phase=doc.get("phase"),
        microbatches=doc.get("microbatches"),
        gcs_root=doc.get("gcs_root"),
        segment=doc.get("segment"),
        state=doc.get("state"),
        is_frontier=bool(doc.get("isFrontier", True)),
        closes_segment=bool(doc.get("closesSegment", False)),
        corroboration_only=bool(doc.get("corroborationOnly", False)),
        expected_runtime_hours=doc.get("expectedRuntimeHours"),
        proof=_parse_proof(doc.get("proof")),
    )


def result_bundle(*, outcome: Outcome, receipt: dict) -> dict:
    """The submitted document, in the API's own field names.

    Built from the receipt already written to disk, so what is sent and what is
    kept cannot diverge. `resultJson` is audit_replay's own output verbatim —
    the record wants the primary artifact, not our summary of it.
    """
    losses = receipt.get("losses") or []
    final = losses[-1] if losses else {}
    body = {
        "outcome": outcome.value,
        "reportedStateHash": receipt.get("reproduced_hash"),
        "hardware": receipt.get("machine"),
        "runtimeHours": round((receipt.get("runtime_seconds") or 0) / 3600, 4),
        # Filed under the record's own numbering. `until_step` is the log
        # number, one higher, and submitting that would address a different
        # step than the claim was issued for.
        "step": receipt.get("audit_step", receipt.get("until_step")),
    }
    # A match must carry the losses: the committed hash is public, so echoing it
    # proves nothing, and the API refuses a match without them.
    if final:
        body["ce"] = final.get("loss_ce")
        body["zLoss"] = final.get("loss_zloss")
    if digest := (receipt.get("artifact") or {}).get("digest"):
        body["artifactDigest"] = digest
    if raw := receipt.get("result_json"):
        body["resultJson"] = raw[:16_384]
    return body
