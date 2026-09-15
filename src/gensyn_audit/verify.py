"""Three different questions about a predecessor, kept apart.

An auditor about to spend half a day replaying from someone else's checkpoint
wants to know three things, and conflating any two of them is how a tool ends
up saying "verified" about something it never checked:

1. **Artifact integrity** — are these the bytes the record published? A
   BLAKE2b-256 bundle digest (or a directory's per-file SHA-256 manifest)
   the *record* serves, checked against the download before
   anything opens it. The run's committed ``state_hash`` is not this: it is a
   fact about the training state, and rewriting a checkpoint's tensors leaves
   whatever ``state_hash.txt`` says untouched.
2. **Tensor-state commitment** — is the state inside these bytes the state the
   run committed to at that step? That is the v3 chained hash, and
   reconstructing it needs the target step's gradients and every
   rank's batch chain. Runs in the kit's pinned pretrain, on disk, in seconds.
3. **The replay** — does the next interval reproduce its own target? Hours, and
   the reason the first two exist.

Which of them apply depends on where the predecessor came from, and *only the
record gets to say*:

``gensyn-anchor``
    An original Gensyn training checkpoint. These predate the gradient sidecar
    and carry no gradients, so (2) cannot be reconstructed for them and is
    explicitly skipped — an assumption about the starting checkpoint, reported
    as one. (1) applies when the record publishes a manifest; otherwise the
    anchor is trusted by provenance and its bytes are reported as unverified.

``crowd``
    Another auditor's hand-off. (1) and (2) both apply and both fail closed.
    Missing gradients, a missing rank chain, forged metadata or a hash mismatch
    stop the audit; none of them fall back to the anchor path.

``unknown``
    The record said something this tool does not recognise, or said nothing.
    Fails closed. A predecessor that cannot be classified cannot be gated, and
    guessing is how the anchor path becomes a bypass.

Nothing here infers provenance from the artifact: not from the URI's host, not
from a flag inside ``meta.json``, and not from the absence of a gradients file.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import convert as convertmod
from . import handoff as handoff_mod
from . import record as recordmod
from .errors import AuditError
from .kit import KitFile, _digest, verify_entrypoint
from .ui import PASS, Activity, echo, kv, paint
from .upload import digest_file

ANCHOR_REPORT = "Trusted Gensyn anchor"


@dataclass(frozen=True)
class Verdict:
    """What was actually established, in words that survive being quoted."""

    provenance: str
    artifact: str
    """``verified`` | ``unverified``; failures raise AuditError."""

    state_hash: str
    """``verified`` | ``skipped-anchor``; failures raise AuditError."""

    checkpoint: Path | None = None
    """The verified directory to replay, including a restored packed handoff."""

    bundle: Path | None = None
    """The packed hand-off the gate digest-checked and unpacked, when there
    was one. Nothing reads it after the gate; `Workdir.consume_predecessor_bundle`
    decides whether it is this workdir's to delete."""

    bundle_digest: str | None = None
    """The BLAKE2b-256 the bundle was held to -- the record's, confirmed
    against the bytes. Saved beside the verdict so a `--restart` can tell that
    `verified_predecessor` still came from the bytes the record names."""

    @property
    def summary(self) -> str:
        if self.provenance == recordmod.ANCHOR:
            return ANCHOR_REPORT
        return (
            "Crowd hand-off; artifact integrity verified; training-state hash "
            "reconstructed and matched against the published log."
        )

    def record(self) -> dict:
        """The block written into the receipt, so a reader of the result can
        tell which of the three checks actually ran."""
        return {
            "predecessor_provenance": self.provenance,
            "artifact_integrity": self.artifact,
            "tensor_state_commitment": self.state_hash,
            "statement": self.summary,
        }


# ── 1. artifact integrity ────────────────────────────────────────────────────


def check_artifact(root: Path, listed: tuple[KitFile, ...]) -> None:
    """Hold every byte of the download to the digests the record published.

    Every listed file must be present with its sha256, and the directory must
    hold nothing else: a checkpoint is loaded by name, so an unlisted extra
    file is as much a change to the artifact as an edited one. Same check
    ``kit.stage`` applies to the wheels, for the same reason.
    """
    by_name = {f.name: f for f in listed}
    if not listed or len(by_name) != len(listed):
        raise AuditError("the artifact manifest is empty or contains duplicate names.")
    members = [root] if root.is_file() else list(root.rglob("*"))
    base = root.parent if root.is_file() else root
    if root.is_symlink() or any(p.is_symlink() for p in members):
        raise AuditError("checkpoint artifacts must not contain symbolic links.")
    on_disk = {str(p.relative_to(base)) for p in members if p.is_file()}

    if missing := sorted(set(by_name) - on_disk):
        raise AuditError(
            f"the record publishes {len(missing)} file(s) this download does "
            f"not have: {', '.join(missing[:4])}.",
            hint="The transfer is incomplete. Re-fetch; do not replay from it.",
        )
    if extra := sorted(on_disk - set(by_name)):
        raise AuditError(
            f"this checkpoint carries {len(extra)} file(s) the record does not "
            f"publish: {', '.join(extra[:4])}.",
            hint="Every file in a published checkpoint is named in its "
            "manifest. An unlisted one was not put there by the run.",
        )
    for name, entry in sorted(by_name.items()):
        got = _digest(base / name)
        if got != entry.sha256:
            raise AuditError(
                f"{name} is not the file the record published.\n"
                f"    downloaded  {got}\n"
                f"    record      {entry.sha256}",
                hint="Refusing to open it. This is the check that runs before "
                "the pickle in dcp/.metadata does, so it is not one to "
                "skip and re-examine afterwards.",
            )


# ── 2. tensor-state commitment ───────────────────────────────────────────────


def check_state_hash(
    venv: Path, checkpoint: Path, *, expect: str, prev: str, device: str = "cpu"
) -> dict:
    """Reconstruct the checkpoint's published v3 hash, in the kit's pretrain.

    The heavy half — building the model, loading DCP, re-slicing the state into
    the recorded shard layout — is the pinned implementation's, invoked as a
    subprocess in the audit venv. Reimplementing it here would mean two
    definitions of the hash the whole record rests on, differing quietly.
    """
    entry = verify_entrypoint(venv)
    if not entry.is_file():
        raise AuditError(
            f"the audit venv has no {entry.name}.",
            hint="Install the selected kit with `gensyn-audit install` first. "
            "If the executable is still missing, use a kit containing the "
            "handoff verifier. A crowd hand-off must not be replayed from unchecked.",
        )
    proc = subprocess.run(
        [
            str(entry),
            "--checkpoint",
            str(checkpoint),
            "--expect-hash",
            expect,
            "--prev-hash",
            prev,
            "--device",
            device,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        result = json.loads(proc.stdout)
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuditError(
            "the hand-off verifier produced no result.",
            hint=(proc.stderr or proc.stdout).strip()[-900:],
        ) from exc
    if not isinstance(result, dict):
        raise AuditError("the hand-off verifier returned a non-object result.")
    if proc.returncode != 0:
        raise AuditError(
            f"the hand-off verifier failed (exit {proc.returncode}).",
            hint=str(result.get("error") or proc.stderr or proc.stdout)[-900:],
        )
    if result.get("verified") is not True:
        raise AuditError(
            "this hand-off is not the state the run committed to.\n"
            f"    reconstructed  {result.get('reconstructed_hash', '—')}\n"
            f"    published      {expect}",
            hint=(
                result.get("error")
                or "The bytes loaded fine; the state inside them is not the one "
                "the run published for this step. Nothing here is worth a "
                "replay, and it is worth reporting rather than re-fetching."
            ),
        )
    if (
        result.get("reconstructed_hash") != expect
        or result.get("expected_hash") != expect
        or result.get("prev_hash") != prev
    ):
        raise AuditError("the verifier's result does not match the requested commitments.")
    world = result.get("dp_world_size")
    if type(world) is not int or not 1 <= world <= handoff_mod.MAX_WORLD:
        raise AuditError("the verifier returned an invalid DP world size.")
    return result


# ── the gate ─────────────────────────────────────────────────────────────────


def _commitments(source: str | None, why: str):
    if not source:
        raise AuditError(
            f"no published state-hash log, so {why}.",
            hint="Name it in the manifest as `artifacts.state_hashes`, or pass "
            "--state-hashes. It has to be the run's own published log: a "
            "hash supplied by whoever produced the artifact is that "
            "author's word about their own work.",
        )
    from . import commitments as commitmentsmod

    return commitmentsmod.load(source)


def gate(
    *,
    checkpoint: Path,
    pred,
    venv: Path,
    state_hashes: str | None,
    device: str = "cpu",
    unpack_to: Path | None = None,
) -> Verdict:
    """Decide, and either return a verdict or refuse. Runs before the replay.

    ``pred`` is the record's ``Predecessor``; it alone decides which path this
    takes. ``state_hashes`` names the run's published log, which supplies both
    hashes a crowd hand-off is checked against.
    """
    if not checkpoint.exists():
        raise AuditError(
            f"no checkpoint directory at {checkpoint}.",
            hint="Nothing downloaded here yet. `run` fetches the predecessor, or "
            "pass --checkpoint with a directory you already hold.",
        )
    provenance = pred.provenance

    if provenance == recordmod.UNKNOWN:
        raise AuditError(
            f"the record does not say what this step's predecessor is (kind={pred.source!r}).",
            hint="A predecessor that cannot be classified cannot be gated: an "
            "original Gensyn anchor skips the state-hash reconstruction, "
            "a crowd hand-off must pass it, and defaulting to the "
            "forgiving one would make the anchor path a bypass. The "
            "record has to label it.",
        )

    if provenance == recordmod.ANCHOR:
        if not checkpoint.is_dir():
            raise AuditError("a trusted anchor must be a checkpoint directory.")
        if pred.artifact_files:
            check_artifact(checkpoint, pred.artifact_files)
            verdict = Verdict(provenance, "verified", "skipped-anchor")
        else:
            verdict = Verdict(provenance, "unverified", "skipped-anchor")
        echo(kv("predecessor", verdict.summary))
        return verdict

    # Crowd handoffs must pass integrity checks before invoking a loader.
    bundle = checkpoint if checkpoint.is_file() else checkpoint / convertmod.BUNDLE_NAME
    bundle_digest = None
    if bundle.is_file() and pred.digest:
        if not pred.has_bundle_digest:
            raise AuditError("predecessor.digest must be 64 hexadecimal characters.")
        if checkpoint.is_symlink() or bundle.is_symlink():
            raise AuditError("checkpoint artifacts must not contain symbolic links.")
        with Activity("checking the bundle digest", detail="BLAKE2b over the whole file"):
            got = digest_file(bundle)
        if got != pred.digest:
            raise AuditError(
                "the bundle is not the file the record published (BLAKE2b-256 mismatch).",
                hint="Re-fetch the hand-off; refusing to unpack it.",
            )
        bundle_digest = got
        # Only the authenticated bundle is consumed; siblings are not loaded.
        if pred.artifact_files:
            check_artifact(checkpoint, pred.artifact_files)
    elif pred.artifact_files:
        check_artifact(checkpoint, pred.artifact_files)
    else:
        raise AuditError(
            "the record publishes no usable artifact digest for this crowd hand-off.",
            hint="Packed hand-offs require predecessor.digest (BLAKE2b-256); "
            "directories require artifactFiles. Tensor-state verification "
            "does not replace pre-load integrity checks.",
        )
    echo(f"  {paint(PASS, 'green')} artifact matches the digests the record publishes")
    if type(pred.step) is not int or pred.step <= 0:
        raise AuditError("the record must declare a positive predecessor step.")

    packed = bundle if bundle.is_file() else None
    if packed is not None:
        if unpack_to is None:
            raise AuditError("a packed handoff needs a separate unpack destination.")
        with Activity(
            "unpacking the hand-off",
            detail="the converter verifies every tensor it writes; minutes at 18 GB",
        ):
            checkpoint = convertmod.unpack(venv, bundle, unpack_to)

    handoff = handoff_mod.inspect(checkpoint)
    handoff_mod.require_complete(handoff, what="this crowd hand-off")
    if handoff.step != pred.step:
        raise AuditError("the hand-off's step does not match the record's predecessor.")

    log = _commitments(state_hashes, "a crowd hand-off cannot be verified")
    expect = log.require(pred.step)
    prev_step, prev = log.previous(pred.step)
    echo(kv("published hash", expect, f"log step {pred.step}"))
    echo(kv("chains from", prev[:16] + "…", f"log step {prev_step}"))

    with Activity(
        "reconstructing the state hash",
        detail="the kit's verifier loads the checkpoint and re-derives it",
    ):
        result = check_state_hash(venv, checkpoint, expect=expect, prev=prev, device=device)
    if type(result.get("step")) is not int or result["step"] != pred.step:
        raise AuditError("the verifier returned the wrong checkpoint step.")
    echo(
        f"  {paint(PASS, 'green')} tensor state reconstructed and matched "
        f"({result['dp_world_size']} DP rank(s), gradients from the sidecar)"
    )
    verdict = Verdict(
        recordmod.CROWD,
        "verified",
        "verified",
        checkpoint,
        bundle=packed,
        bundle_digest=bundle_digest,
    )
    echo(kv("predecessor", verdict.summary))
    return verdict
