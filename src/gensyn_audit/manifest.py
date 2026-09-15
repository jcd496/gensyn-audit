"""The run manifest: where everything lives, so nothing is compiled in.

A kit pins the *toolchain*; a trajectory pins the *hashes*. Neither says where
this run's checkpoints, shards or record are, and hardcoding a bucket name into
a verification tool is how an auditor ends up unable to check a run that moved.

So the manifest is the one document that names locations, and it is a static
file because none of them change once a run is published. Everything the CLI
resolves -- the kit prefix, the record's base URL, the checkpoint and shard
roots -- comes from here or from an explicit flag, and from nowhere else.

This is the ``--manifest`` the web app hands you, and it makes the command the
record prints literally correct::

    gensyn-audit run --run <id> --step <n> --claim <token> --manifest <url>

Schema (v1)::

    {
      "manifest_format": 1,
      "run": "20260703-171943-4e85cd3",
      "kit": "gs://<bucket>/audit-kit/pt-<sha12>_rp-<sha12>",
      "record": "https://<host>",
      "artifacts": {
        "checkpoints": "gs://<bucket>/<path>/checkpoints",
        "shards":      "gs://<bucket>/<path>/data/shards",
        "state_hashes":"gs://<bucket>/<path>/logs/state_hashes.jsonl"
      }
    }

Every field but ``run`` is optional: a flag beats the manifest, and a manifest
that names only a kit is perfectly valid for verifying init units, which
download nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .errors import AuditError
from .kit import _read  # one reader for gs:// / https:// / local, already tested

MANIFEST_FORMAT = 1


@dataclass(frozen=True)
class Artifacts:
    """Where this run's bytes live. All optional; init units need none of them."""

    checkpoints: str | None = None
    shards: str | None = None
    state_hashes: str | None = None

    def checkpoint_uri(self, step: int | None) -> str | None:
        """``<checkpoints>/step_000000100`` — the loop's own directory naming.

        `step` is None whenever the record does not know the predecessor yet:
        a position-2 step whose hand-off nobody has produced carries no
        `predecessor.step` at all. There is no path to derive, and formatting
        None as `:09d` raised a TypeError over the message that explains the
        wait.
        """
        if not self.checkpoints or step is None:
            return None
        return f"{self.checkpoints.rstrip('/')}/step_{step:09d}"


@dataclass(frozen=True)
class Manifest:
    run: str
    kit: str | None
    record: str | None
    artifacts: Artifacts
    source: str

    def require_kit(self) -> str:
        if not self.kit:
            raise AuditError(
                f"manifest {self.source} names no kit.",
                hint="Add a `kit` field, or pass --kit <prefix>.",
            )
        return self.kit


def load(source: str) -> Manifest:
    raw = _read(source)
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuditError(f"manifest {source} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise AuditError(f"manifest {source} must be a JSON object.")

    if _is_record_manifest(doc):
        return _from_record_manifest(doc, source)

    fmt = doc.get("manifest_format", MANIFEST_FORMAT)
    if fmt != MANIFEST_FORMAT:
        raise AuditError(
            f"manifest {source} declares manifest_format {fmt}; this runner understands "
            f"{MANIFEST_FORMAT}.",
            hint="pip install -U gensyn-audit",
        )
    if not doc.get("run"):
        raise AuditError(f"manifest {source} has no `run` id.")

    art = doc.get("artifacts") or {}
    if not isinstance(art, dict):
        raise AuditError(f"manifest {source}: `artifacts` must be an object.")

    return Manifest(
        run=str(doc["run"]),
        kit=doc.get("kit") or None,
        record=doc.get("record") or None,
        artifacts=Artifacts(
            checkpoints=art.get("checkpoints") or None,
            shards=art.get("shards") or None,
            state_hashes=art.get("state_hashes") or None,
        ),
        source=source,
    )


def _is_record_manifest(doc: dict) -> bool:
    """The record serves its own `/manifest.json`, and it is not this schema.

    The web app prints `--manifest <record>/manifest.json`, so that URL has to
    work. It is told apart by shape, not by a version field: the record's
    document nests the run under `run` and has no `manifest_format`.
    """
    return "manifest_format" not in doc and isinstance(doc.get("run"), dict)


def _from_record_manifest(doc: dict, source: str) -> Manifest:
    """Map the record's manifest onto ours.

    The record names the kit and, being the record, is its own record URL. It
    does not yet name artifact roots, so interval units still need them passed
    explicitly -- that gap is upstream, and `Artifacts` stays empty rather than
    guessing a bucket.
    """
    run = doc.get("run") or {}
    audit = doc.get("audit") or {}
    kit = audit.get("kit")
    kit_prefix = None
    if isinstance(kit, dict):
        kit_prefix = kit.get("urlBase") or None
        if not kit_prefix and kit.get("manifestUrl"):
            kit_prefix = str(kit["manifestUrl"]).rsplit("/", 1)[0]
    elif isinstance(kit, str):
        kit_prefix = kit

    # Endpoints address the run by NAME (`/v1/runs/open-1b/...`), not by id.
    name = run.get("name") or run.get("id")
    if not name:
        raise AuditError(f"manifest {source} names no run.")

    art = doc.get("artifacts") or {}
    return Manifest(
        run=str(name),
        kit=kit_prefix,
        record=_origin(source),
        artifacts=Artifacts(
            checkpoints=art.get("checkpoints") or None,
            shards=art.get("shards") or None,
            state_hashes=art.get("state_hashes") or None,
        ),
        source=source,
    )


def _origin(url: str) -> str | None:
    """`https://host/manifest.json` -> `https://host`."""
    m = re.match(r"^(https?://[^/]+)", url)
    return m.group(1) if m else None


def resolve(source: str | None) -> Manifest | None:
    """Load a manifest if one was given. `None` is not an error: every value it
    would supply can also be passed as a flag."""
    if not source:
        return None
    return load(source)
