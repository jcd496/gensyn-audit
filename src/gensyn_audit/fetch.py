"""Pulling the predecessor checkpoint, and refusing to replay the wrong one.

DCP metadata uses pickle, so checkpoints are untrusted input. The Open 1B
manifest pins a loader that validates metadata with an allowlisted unpickler
before calling `dcp.load`, and `verify.py` checks record-published artifact
digests before invoking that loader.

This module downloads the predecessor and performs the cheapest preliminary
check: whether `state_hash.txt` names the expected step. That catches a stale or
wrong directory, but the file is part of the artifact and is not evidence by
itself. Artifact integrity and the training-state commitment are checked in
`verify.py` immediately afterward.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlsplit

from .errors import AuditError
from .kit import Unit, local_uri_path
from .ui import ARROW, echo, paint
from .ui import PASS as PASS_MARK


def _transfer_block():
    """The live block a download shows while it moves: bar, bytes, rate, estimate.

    Shared by both download paths rather than written once for the directory
    one. A checkpoint directory is a few hundred objects and a packed handoff
    is a single multi-gigabyte one, but what the auditor needs to see is the
    same either way, and a path that shows nothing reads as a hang. Returns the
    Live handle so the caller can clear it, and the callback to hand to gcs.
    """
    import time

    from .ui import Live, bar, human_bytes, human_duration

    live = Live()
    started = time.monotonic()

    def on_progress(
        done: int, total: int, index: int = 1, count: int = 1, current: str = ""
    ) -> None:
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = done / elapsed
        # Remaining time from the rate achieved so far. Honest for a transfer
        # this long: it settles within seconds and does not pretend to know
        # more than it does.
        left = human_duration((total - done) / rate) if rate > 0 else "?"
        # Pad the plain text, then colour: an f-string width counts escape
        # bytes, which silently shortens the column.
        label = paint(f"{'downloading':<22}", "cyan")
        # Keep the END of the name. Objects in one prefix share a long head
        # (`LC08_L1GT_044034_20130330_…`, `dcp/__…`), so a head-truncated label
        # is identical for every file and says nothing about which is in flight.
        short = current if len(current) <= 38 else "…" + current[-37:]
        where = paint(short if count == 1 else f"{index}/{count} objects · {short}", "dim")
        pct = int(100 * done / total) if total else 0
        detail = (
            f"{human_bytes(done)} / {human_bytes(total)} · {human_bytes(rate)}/s · ~{left} left"
        )
        live.update(
            [
                f"  {label} {where}",
                f"  {bar(done / total if total else None)} {pct:3d}%  {paint(detail, 'dim')}",
            ]
        )

    return live, on_progress


def _copy_tree(uri: str, dest: Path, *, project: str | None, quiet: bool) -> None:
    """Mirror a checkpoint directory locally, resumably and verified.

    Roughly 19 GB over whatever connection the auditor has, so it gets the same
    live block the replay does. Without one this is twenty silent minutes, and
    silence during a large transfer reads as a hang -- especially with the work
    concentrated in a few multi-gigabyte shards.
    """
    from . import gcs
    from .ui import human_bytes

    del project, quiet
    dest.mkdir(parents=True, exist_ok=True)

    live, on_progress = _transfer_block()
    try:
        total = gcs.download_prefix(uri, dest, on_progress=on_progress)
    finally:
        live.clear()
    echo(f"  {paint(PASS_MARK, 'green')} {human_bytes(total)} fetched and checksum-verified")


def _copy_file(uri: str, dest: Path, *, project: str | None, progress: bool = False) -> None:
    """One object to `dest`, with the live block only where it is worth it.

    fetch_descriptor pulls a meta.json of a few kilobytes and prints its own
    line when it lands, so it leaves `progress` off. A packed handoff is the
    entire predecessor state in one object, and without the block that path
    printed nothing at all while it moved gigabytes: no bar, no byte count, not
    even the URI it was reading. That is the shape of the only bug an auditor
    can do nothing about, because it is indistinguishable from a hang.

    No verification claim on this path, unlike the directory one. gcs.download
    checks whatever checksum the response carried, but a packed artifact's
    authentication is the shared gate's job before unpacking, so this says only
    what it did: bytes arrived.
    """
    from . import gcs
    from .ui import human_bytes

    del project
    if not progress:
        gcs.download(uri, dest)
        return

    name = uri.rstrip("/").rsplit("/", 1)[-1]
    live, on_progress = _transfer_block()
    try:
        gcs.download(
            uri, dest, on_progress=lambda done, total: on_progress(done, total, 1, 1, name)
        )
    finally:
        live.clear()
    echo(f"  {paint(PASS_MARK, 'green')} {human_bytes(dest.stat().st_size)} fetched")


def read_state_hash(checkpoint: Path) -> str | None:
    """The digest the checkpoint claims for itself."""
    path = checkpoint / "state_hash.txt"
    if not path.is_file():
        return None
    return path.read_text().strip().lower()


def verify_predecessor(checkpoint: Path, unit: Unit, expected: str | None) -> None:
    """Does this checkpoint even claim to be the step the record named?

    Deliberately the weakest of the three checks, and labelled as such. It
    compares the checkpoint's own ``state_hash.txt`` -- written by whoever
    produced it -- against the hash the record committed for that step. That
    catches the wrong download and a stale directory, which is worth catching
    for the price of reading 65 bytes.

    It is NOT an artifact digest: the file is inside the artifact, so rewriting
    a checkpoint's tensors leaves it saying whatever its author wants. And it is
    not the training-state commitment either, which is reconstructed from the
    tensors themselves. Both of those live in `verify.py`, which runs after
    this.
    """
    # A packed artifact's record digest is not a training-state commitment.
    # Its bytes are authenticated by the shared gate before unpacking.
    if checkpoint.is_file() or (checkpoint / "handoff.safetensors").is_file():
        return
    got = read_state_hash(checkpoint)

    if expected is None:
        return

    if got is None:
        raise AuditError(
            f"{checkpoint} has no state_hash.txt, so it cannot be checked against the "
            f"record's commitment ({expected}).",
            hint="Either the download is incomplete (re-run `audit fetch`) or this is not "
            "an auditable checkpoint. Do not replay it.",
        )

    if got != expected:
        raise AuditError(
            f"starting checkpoint is not the step the record named.\n"
            f"    on disk    {got}\n"
            f"    committed  {expected}",
            hint="These are different states, so there is nothing here to audit. "
            "Delete the directory and re-fetch; if it happens again the record "
            "and the storage disagree, which is worth reporting rather than "
            "working around.",
        )

    echo(f"  {paint(PASS_MARK, 'green')} starting checkpoint declares the step the record named")


def fetch_checkpoint(
    uri: str,
    dest: Path,
    unit: Unit,
    *,
    expected_digest: str | None = None,
    project: str | None = None,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Ensure the verified predecessor checkpoint is at `dest`. Idempotent.

    Takes the URI and the expected digest explicitly rather than an entry
    object: they come from the step API, which the trajectory format does not
    yet carry, and passing them separately keeps this honest about that.
    """
    if local_uri_path(uri) is not None:
        if not dest.exists():
            raise AuditError(
                f"the unit names a local checkpoint at {dest}, which does not exist.",
                hint="Pass --checkpoint <dir> to point at one you already hold.",
            )
        echo(f"  using {dest}")
        verify_predecessor(dest, unit, expected_digest)
        return dest

    if force and dest.exists():
        if dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)

    if urlsplit(uri).path.endswith(".safetensors"):
        dest.mkdir(parents=True, exist_ok=True)
        bundle = dest / "handoff.safetensors"
        echo(f"  {uri} {ARROW} {bundle}")
        _copy_file(uri, bundle, project=project, progress=True)
        return bundle

    already = read_state_hash(dest)
    if already and expected_digest and already == expected_digest:
        # Fast path only when the record published a digest and the bytes on
        # disk claim it. Without one, `state_hash.txt` proves nothing about
        # completeness -- it is a 65-byte file that lands early in a 19 GB
        # transfer -- and `verify_predecessor` cannot help either without an
        # expected digest. So fall through: `download_prefix`
        # skips every object already present at the right size, which costs one
        # listing instead of the download, and is an actual completeness check.
        echo(f"  {paint(PASS_MARK, 'green')} checkpoint already present and verified")
        return dest

    echo(f"  {uri} {ARROW} {dest}")
    _copy_tree(uri, dest, project=project, quiet=quiet)
    verify_predecessor(dest, unit, expected_digest)
    return dest


def fetch_descriptor(uri: str | None, dest: Path, *, project: str | None = None) -> Path | None:
    """A segment boundary's descriptor: ``meta.json`` and nothing else.

    audit_replay reads only that one file from --descriptor-checkpoint, so
    there is no reason to pull another 18 GB for it.
    """
    if not uri:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "meta.json"
    local = local_uri_path(uri)
    if local is not None:
        source = local / "meta.json"
        if not source.is_file():
            raise AuditError(f"descriptor checkpoint {local} has no meta.json.")
        shutil.copyfile(source, target)
    else:
        _copy_file(uri.rstrip("/") + "/meta.json", target, project=project)
    echo(f"  {paint(PASS_MARK, 'green')} segment descriptor {ARROW} {target}")
    return dest
