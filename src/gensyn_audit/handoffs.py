"""The hand-offs this machine has packed, so it never downloads its own upload.

An auditor who takes two steps in a row uploads a hand-off, and the record
then names that same file as the next step's predecessor. Without this index
the CLI fetched it straight back -- ~18 GB, over the same residential link that
had just sent it -- to arrive at bytes already sitting in the previous workdir.

Entries are keyed by the BLAKE2b-256 digest the record publishes, which is the
only thing that makes a local file eligible: the path is a *hint about where
those bytes might be*, and `verify.gate` still holds whatever it finds there to
the record's digest before anything opens it. A stale or edited file fails that
gate exactly as a corrupt download would.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .plan import cache_root

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Known:
    path: Path
    digest: str
    size: int
    step: int | None
    run: str | None


def _index_dir() -> Path:
    return cache_root() / "handoffs"


def _entry_path(digest: str) -> Path:
    return _index_dir() / f"{digest}.json"


def remember(path: Path, digest: str, *, step: int | None = None, run: str | None = None) -> None:
    """Record where a packed bundle with this digest lives. Best effort: an
    index that cannot be written costs a download, not an audit."""
    if not _DIGEST.match(digest or ""):
        return
    try:
        stat = path.stat()
        _index_dir().mkdir(parents=True, exist_ok=True)
        _entry_path(digest).write_text(
            json.dumps(
                {
                    "path": str(path.resolve()),
                    "digest": digest,
                    "size": stat.st_size,
                    "step": step,
                    "run": run,
                },
                indent=2,
            )
            + "\n"
        )
    except OSError:
        return


def find(digest: str | None) -> Known | None:
    """A local bundle recorded under this digest, if it is still where it was.

    Only the cheap facts are checked here -- the file exists and is the size it
    was when packed. The digest itself is the gate's job, and it runs on every
    predecessor regardless of where the bytes came from.
    """
    if not digest or not _DIGEST.match(digest):
        return None
    entry = _entry_path(digest)
    try:
        doc = json.loads(entry.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or doc.get("digest") != digest:
        return None
    path = Path(str(doc.get("path", "")))
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file() or stat.st_size != doc.get("size"):
        return None
    step = doc.get("step")
    return Known(
        path=path,
        digest=digest,
        size=stat.st_size,
        step=step if isinstance(step, int) else None,
        run=doc.get("run") if isinstance(doc.get("run"), str) else None,
    )
