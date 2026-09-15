"""Handing the artifact to the record, resumably.

The upload is ~18 GB on residential upstream, which takes longer than some
replays. So it has to survive an interruption: the bytes go direct to storage
in chunks, and a resumed upload asks where the last one stopped rather than
starting over. Losing a day's replay to a dropped Wi-Fi connection at the last
hop would be the single most demoralising failure this tool could have.

What must travel is the COMPLETE hand-off (see `handoff.py`): the checkpoint,
every rank's batch-hasher chain, and `gradients.safetensors`. A recipient
without those can load the state but cannot reconstruct the hash the run
published for it, which is the whole point of handing it over. So this module
refuses to send a partial hand-off rather than putting one in intake that the
next auditor will have to trust on the uploader's word.

The converter embeds gradients and resume files in the tensor bundle. Packing
happens before submission so its digest describes the bytes actually uploaded;
recipients authenticate those bytes before unpacking and checking tensor state.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import handoff as handoff_mod
from .errors import AuditError
from .record import UploadTicket


@dataclass
class Bundle:
    path: Path
    digest: str
    size: int
    workdir: Path | None = None
    """Where the upload's session record lives between runs. Defaults to the
    bundle's own directory when the caller has nothing better."""


def sidecar(result, bundle: Bundle, *, run_id: str | None = None) -> dict:
    """The ``handoff.json`` the verifier reads beside the bundle.

    Every key here is one the verifier requires and checks: ``step`` against the
    queue item's step, ``state_hash`` and ``bundle_digest`` against what was
    declared at submit, ``run_id`` against its own, ``match`` for true. So the
    values come from the receipt the tool already wrote, not from a second
    look at anything, and ``step`` is the record's number: the queue item is
    filed under it, and ``until_step`` is one higher.

    ``run_id`` is the record's run *id* (``20260722-213626-ad3276b``), which is
    what the verifier is configured with. The receipt's ``run`` is whatever the
    record was addressed by, and the API answers to the run's *name*
    (``open-1b``) as well, so a manifest-driven run carries the name there.
    The first production hand-off was rejected ``bundle_invalid`` for exactly
    that: a sidecar saying ``open-1b`` to a verifier expecting the id.
    """
    last = result.losses[-1] if result.losses else {}
    return {
        "run_id": run_id or result.run,
        "step": result.audit_step,
        "state_hash": result.reproduced_hash,
        "expected_state_hash": result.committed_hash,
        "match": result.matched is True,
        "ce": last.get("loss_ce"),
        "z_loss": last.get("loss_zloss"),
        "bundle_digest": bundle.digest,
        "repop": {"commit": result.repop_commit, "backends": list(result.repop_backends)},
        "device": result.device,
        "produced_by": result.tool,
    }


def send_sidecar(payload: dict, ticket: UploadTicket) -> None:
    """PUT ``handoff.json``. A kilobyte, so no session and no resume."""
    if not ticket.sidecar_url:
        raise AuditError(
            "the record offered no URL for handoff.json.",
            hint="The verifier requires the sidecar beside the bundle and rejects a "
            "submission without it, so this upload cannot be accepted. The bundle "
            "is in intake; the record needs to return `sidecar.signedUrl`.",
        )
    body = json.dumps(payload, sort_keys=True).encode()
    if ticket.sidecar_url.startswith("file://"):
        target = Path(ticket.sidecar_url[len("file://") :])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return
    req = urllib.request.Request(ticket.sidecar_url, data=body, method="PUT")
    req.add_header("content-type", "application/json")
    req.add_header("Content-Length", str(len(body)))
    try:
        urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as exc:
        raise AuditError(
            f"could not upload handoff.json ({exc.code}).",
            hint=exc.read().decode(errors="replace")[:300],
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AuditError(
            f"could not reach the sidecar upload URL: {exc}",
            hint="Re-run the same command: the bundle is not re-sent.",
        ) from exc


#: The record wants blake2b-256 of `handoff.safetensors`, declared at submit
#: time as `artifactDigest` and checked against the bytes after they land.
def digest_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_bundle(source: Path, dest: Path) -> Bundle:
    """Digest the packed file already declared in the submission receipt.

    ``dest`` is the workdir: the bundle itself stays where the converter wrote
    it, but the resumable session opened for it is recorded there.
    """
    if source.is_dir():
        handoff_mod.require_complete(handoff_mod.inspect(source), what="this hand-off")
        source = source / "handoff.safetensors"
    if not source.is_file():
        raise AuditError(
            f"no packed handoff at {source}.",
            hint="Pack the complete checkpoint with the kit's converter before "
            "submission. The replay result is still recorded without an upload.",
        )
    return Bundle(source, digest_file(source), source.stat().st_size, workdir=dest)


def start_session(ticket: UploadTicket) -> str:
    """Open the GCS resumable session and return the URI to PUT to.

    The signed URL only *starts* a session: POST it with
    ``x-goog-resumable: start``, and the session URI comes back in ``Location``.
    PUTting the bundle at the signed URL directly would upload it as a single
    unresumable request, which for 16-18 GB on a home connection is the one
    thing that must not happen.
    """
    if ticket.signed_url.startswith("file://"):
        return ticket.signed_url
    req = urllib.request.Request(ticket.signed_url, data=b"", method="POST")
    req.add_header("x-goog-resumable", "start")
    req.add_header("content-type", "application/octet-stream")
    req.add_header("Content-Length", "0")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            location = resp.headers.get("Location")
    except urllib.error.HTTPError as exc:
        raise AuditError(
            f"could not start the upload session ({exc.code}).",
            hint=exc.read().decode(errors="replace")[:300],
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AuditError(f"could not reach the upload endpoint: {exc}") from exc
    if not location:
        raise AuditError("the upload session returned no Location header.")
    return location


#: Where the session URI is kept between runs. Without it every run opened a
#: fresh session and "re-run to resume" restarted from byte 0.
SESSION_FILE = ".upload-session.json"


def _session_path(bundle: Bundle) -> Path:
    return (bundle.workdir or bundle.path.parent) / SESSION_FILE


def _object_of(ticket: UploadTicket) -> str:
    """What a session is for: the object, not the hour-limited signature on it."""
    return ticket.object_uri or ticket.signed_url.split("?", 1)[0]


def _session_identity(bundle: Bundle, ticket: UploadTicket) -> dict:
    """What a saved session must match to be resumed.

    The object alone is not enough: if the bundle was re-packed between
    attempts, resuming at the old committed offset would append bytes of the
    new file to bytes of the old one, and the digest check would reject a
    hybrid nothing on disk ever looked like. The digest and size pin the
    session to the exact bytes it was opened for.
    """
    return {"object": _object_of(ticket), "digest": bundle.digest, "size": bundle.size}


def _load_session(bundle: Bundle, ticket: UploadTicket) -> str | None:
    """The session URI a previous run opened for these same bytes, if any."""
    path = _session_path(bundle)
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(saved, dict):
        return None
    identity = _session_identity(bundle, ticket)
    if any(saved.get(key) != value for key, value in identity.items()):
        return None
    session = saved.get("session")
    return session if isinstance(session, str) and session else None


def _save_session(bundle: Bundle, ticket: UploadTicket, session: str) -> None:
    path = _session_path(bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**_session_identity(bundle, ticket), "session": session}))


def _forget_session(bundle: Bundle) -> None:
    try:
        _session_path(bundle).unlink()
    except FileNotFoundError:
        pass


def _resume_offset(session: str) -> int | None:
    """How much of this upload already landed, asked of the *session*.

    A zero-length PUT with ``Content-Range: bytes */*`` to the session URI
    answers 308 with the committed range, or 200/201 once the object is
    complete. It has to be the session URI: the signed URL is a POST
    signature over ``x-goog-resumable``, and a PUT there is answered 400
    ``MalformedSecurityHeader`` before anything is uploaded, which is how the
    first production upload died with "could not query the upload's progress".

    Returns the byte offset to continue from, -1 when the object is already
    complete, or None when the session is gone (expired, or never valid) and a
    new one has to be opened.
    """
    req = urllib.request.Request(session, method="PUT")
    req.add_header("Content-Length", "0")
    req.add_header("Content-Range", "bytes */*")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status in (200, 201):
                return -1
    except urllib.error.HTTPError as exc:
        if exc.code == 308:  # Resume Incomplete
            rng = exc.headers.get("Range")
            if rng and rng.startswith("bytes=0-"):
                return int(rng.split("-")[1]) + 1
            return 0
        if exc.code in (400, 404, 410):
            return None
        raise AuditError(f"could not query the upload's progress ({exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AuditError(f"could not reach the upload URL: {exc}") from exc
    return 0


def _file_offset(ticket: UploadTicket) -> int:
    """A file:// target answers with its own size."""
    target = Path(ticket.signed_url[len("file://") :])
    return target.stat().st_size if target.is_file() else 0


def send(bundle: Bundle, ticket: UploadTicket, *, on_progress=None) -> None:
    """Upload the bundle, resuming from wherever the last attempt stopped.

    The session URI is what resumes: it is opened once per object and kept in
    the workdir, so a re-run after a dropped connection asks that session how
    far it got rather than opening another and starting over.
    """
    if ticket.signed_url.startswith("file://"):
        offset = _file_offset(ticket)
        if offset >= bundle.size:
            return
        _send_file(bundle, Path(ticket.signed_url[len("file://") :]), offset, on_progress)
        return

    session = _load_session(bundle, ticket)
    offset = _resume_offset(session) if session else None
    if offset is None:
        session = start_session(ticket)
        _save_session(bundle, ticket, session)
        offset = 0
    if offset == -1 or offset >= bundle.size:
        if on_progress:
            on_progress(bundle.size, bundle.size)
        return

    assert session is not None
    with open(bundle.path, "rb") as fh:
        fh.seek(offset)
        while offset < bundle.size:
            chunk = fh.read(ticket.chunk_bytes)
            if not chunk:
                break
            end = offset + len(chunk) - 1
            req = urllib.request.Request(session, data=chunk, method="PUT")
            req.add_header("Content-Length", str(len(chunk)))
            req.add_header("Content-Range", f"bytes {offset}-{end}/{bundle.size}")
            try:
                urllib.request.urlopen(req, timeout=300)
            except urllib.error.HTTPError as exc:
                if exc.code != 308:  # 308 is the normal "keep going"
                    raise AuditError(
                        f"upload failed at byte {offset} ({exc.code}).",
                        hint="Re-run the same command: the transfer resumes from here "
                        "rather than restarting.",
                    ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise AuditError(
                    f"upload interrupted at byte {offset}: {exc}",
                    hint="Re-run the same command to resume.",
                ) from exc
            offset = end + 1
            if on_progress:
                on_progress(offset, bundle.size)


def _send_file(bundle: Bundle, target: Path, offset: int, on_progress) -> None:
    """The file:// path, so the mock exercises chunking and resume for real."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(bundle.path, "rb") as src, open(target, "ab") as dst:
        src.seek(offset)
        while offset < bundle.size:
            chunk = src.read(4 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            dst.flush()
            offset += len(chunk)
            if on_progress:
                on_progress(offset, bundle.size)
