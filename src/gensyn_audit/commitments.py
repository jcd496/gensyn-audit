"""Committed hashes, read from the run's own published log.

The record's design splits authority in two, and the split matters:

* **The commitment is static.** A step's hash was fixed when the run produced
  it and is published as a file -- ``logs/state_hashes.jsonl``, one JSON object
  per step. It is not an API row.
* **The predecessor is mutable.** Which artifact a step starts from depends on
  who has audited what, so only the API can answer it.

A simulated record may derive placeholder hashes and identifies them in
``simulatedParts.commitments``. A replay compared against a placeholder cannot
match. When the manifest names a state-hash log, that file is the authoritative
record of the hashes produced by the run; API commitments are a read-through
convenience.

The receipt also carries an inclusion proof tying that commitment to its
segment's Merkle root. ``verify_inclusion`` checks it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .errors import AuditError

HEX64_LEN = 64

#: Node tags. Without them an inner node is a valid leaf preimage, so a subtree
#: could be presented as one step's commitment.
_LEAF, _INNER = b"\x00", b"\x01"


@dataclass(frozen=True)
class Commitments:
    """`step -> state_hash`, from a published state-hash log."""

    source: str
    by_step: dict[int, str]

    def get(self, step: int) -> str | None:
        return self.by_step.get(step)

    def previous(self, step: int) -> tuple[int, str]:
        """The step the chain at ``step`` folds in, and its hash.

        The chain links HASHED steps, not consecutive ones: with a cadence of
        N the value before step k is step k-N, and with checkpoints saved less
        often than hashes it is not the previous checkpoint either. The log is
        the definition of which steps were hashed, so the answer is simply its
        largest positive entry below ``step``. Step zero is a separate init
        fingerprint, not a link in the periodic chain. A cold start instead
        folds in 32 zero bytes. Without an init record (or step one), an empty
        prefix may be a truncated log, so refuse to guess its starting value.
        """
        if step <= 0:
            raise AuditError("initialization has no preceding periodic commitment.")
        earlier = [s for s in self.by_step if 0 < s < step]
        if not earlier and (0 in self.by_step or step == 1):
            return 0, "0" * HEX64_LEN
        if not earlier:
            raise AuditError(
                f"{self.source} carries no hash before step {step}, so the "
                "value this step's chain folds in is unknown.",
                hint="The chain at the first hashed step folds in nothing; any "
                "later step needs its predecessor's published hash. If the "
                "log starts after this step, it is the wrong log.",
            )
        prev = max(earlier)
        return prev, self.by_step[prev]

    def require(self, step: int) -> str:
        digest = self.by_step.get(step)
        if digest is None:
            known = sorted(self.by_step)
            span = f"{known[0]}–{known[-1]}" if known else "none"
            raise AuditError(
                f"{self.source} carries no hash for step {step}.",
                hint=f"It covers steps {span}. A step is only auditable if the run "
                "hashed it: a digest exists at step k only when "
                "state_hash.every_n_steps divides k.",
            )
        return digest


def load(source: str) -> Commitments:
    """Read a ``state_hashes.jsonl``. Roughly 12 MB for an 80k-step run."""
    from . import gcs

    if source.startswith("gs://"):
        raw = gcs.get(source).decode("utf-8", errors="replace")
    elif source.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(source, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    else:
        path = Path(source.removeprefix("file://"))
        if not path.is_file():
            raise AuditError(f"no state-hash log at {path}.")
        raw = path.read_text(errors="replace")

    by_step: dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated tail is not worth failing over
        digest = str(row.get("state_hash", "")).lower()
        if "step" in row and len(digest) == HEX64_LEN:
            # A resumed run re-logs overlapping steps; the last occurrence is
            # the surviving segment, matching the stitched logs.
            by_step[int(row["step"])] = digest

    if not by_step:
        raise AuditError(
            f"{source} holds no state hashes.",
            hint="Expected one JSON object per line with `step` and `state_hash`.",
        )
    return Commitments(source=source, by_step=by_step)


def reconcile(step: int, *, published: str | None, from_api: str | None, on_conflict=None) -> str:
    """Decide which committed hash to audit against.

    The published log wins. If the API disagrees, that is worth saying out loud
    rather than silently preferring one: it means either the API is still
    deriving placeholders, or two sources that must agree do not — and both are
    things an auditor should hear before spending a day.
    """
    disagree = published and from_api and published != from_api
    if disagree and on_conflict:
        on_conflict(published, from_api)
    chosen = published or from_api
    if not chosen:
        raise AuditError(
            f"no committed hash for step {step}.",
            hint="Neither the record nor a published state-hash log supplied one. "
            "Name the log in the manifest as `artifacts.state_hashes`.",
        )
    return chosen


@dataclass(frozen=True)
class ProofLink:
    """One sibling on the path from a leaf to its segment root."""

    hash: str
    side: str
    """``left`` if the sibling is hashed before the running node, else ``right``."""


@dataclass(frozen=True)
class Proof:
    """A step's inclusion proof, as the receipt serves it."""

    leaf: str
    """The commitment itself, not its leaf node, so it can be compared with the
    hash being audited."""

    root: str
    path: tuple[ProofLink, ...] = ()


def verify_inclusion(proof: Proof) -> bool:
    """Recompute the segment root from the leaf and its sibling chain.

    The construction is the record's, written out in its ``docs/API.md``:
    sha256 over raw bytes, tagged by node kind.

        leaf node    sha256(0x00 || commitment)
        inner node   sha256(0x01 || left || right)

    An odd node rises unchanged instead of pairing with itself, so a path
    carries no link for the levels a node was promoted through and is not the
    same length for every step in a segment.

    What this establishes is that the record agrees with itself. It is not yet
    third-party proof: the root arrives from the same API as the path, and
    becomes evidence only once it is checked against an anchor the record does
    not control. Those are still placeholders.
    """
    try:
        node = sha256(_LEAF + bytes.fromhex(proof.leaf)).digest()
        for link in proof.path:
            sibling = bytes.fromhex(link.hash)
            pair = sibling + node if link.side == "left" else node + sibling
            node = sha256(_INNER + pair).digest()
    except ValueError:
        return False  # a malformed digest is a failed check, not a crash
    return node.hex() == proof.root
