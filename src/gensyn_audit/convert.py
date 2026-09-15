"""Pack and unpack a hand-off, by driving the kit's own converter.

The relay moves one file between auditors, not a directory. ``audit_replay``
writes a ``torch.distributed.checkpoint`` directory; the record's contract
(GEN-2930) is a single ``handoff.safetensors`` plus a ``handoff.json`` sidecar.
``pretrain-dcp-safetensors`` converts between the two, and it ships in the kit's
pretrain wheel, so the code that understands the sharded layout is versioned
with the layout rather than reimplemented here.

Two directions, at two moments:

``pack``    after a matching replay, before the result is submitted. The
            record wants the bundle's digest declared as ``artifactDigest`` at
            submit time, so the file has to exist by then.
``unpack``  after a crowd predecessor is fetched, before the replay. The replay
            takes ``--checkpoint <dir>`` and reads DCP; it has no safetensors
            input, so the file becomes a directory first.

Only a *full* checkpoint directory round-trips. The converter carries the
resume sidecar -- per-rank RNG and batch-hasher state, ``meta.json``,
``global_stream.json``, ``spike_protocol.json`` -- in a second header blob, and
without it a resumed replay draws different data and reports a NO MATCH that
looks exactly like a divergence. Passing ``--keys`` or a bare ``dcp/`` would
produce weight-only transport, so this never does.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from . import handoff
from .errors import AuditError

#: The console script the pretrain wheel installs. Absent from a kit built
#: before the converter landed, which is a named refusal rather than a crash.
ENTRYPOINT = "pretrain-dcp-safetensors"

BUNDLE_NAME = "handoff.safetensors"


def converter(venv_dir: Path) -> Path:
    return venv_dir / "bin" / ENTRYPOINT


def _run(exe: Path, args: list[str], what: str) -> None:
    proc = subprocess.run([str(exe), *args], capture_output=True, text=True, check=False)
    if proc.returncode == 0:
        return
    tail = (proc.stderr or proc.stdout).strip().splitlines()
    raise AuditError(
        f"{what} failed.",
        hint="\n".join(tail[-6:]) or f"{ENTRYPOINT} exited {proc.returncode}",
    )


def _require(venv_dir: Path, direction: str) -> Path:
    exe = converter(venv_dir)
    if exe.is_file():
        return exe
    raise AuditError(
        f"this kit cannot {direction} a hand-off: it has no {ENTRYPOINT}.",
        hint="The converter ships in the pretrain wheel (GEN-2930). A kit built "
        "before it landed carries no way to read or write the safetensors "
        "hand-off the record exchanges.\nUse a kit whose pretrain commit "
        "includes it.",
    )


def pack(venv_dir: Path, checkpoint_dir: Path, dest: Path) -> Path:
    """``step_N/`` -> ``handoff.safetensors``, resume sidecar included."""
    exe = _require(venv_dir, "produce")
    if not checkpoint_dir.is_dir():
        raise AuditError(f"no hand-off checkpoint at {checkpoint_dir}.")
    handoff.require_complete(handoff.inspect(checkpoint_dir), what="this hand-off")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # --verify re-reads the file and compares every tensor bit for bit. This
    # runs once per audit, against bytes the next auditor replays from, so the
    # seconds it costs are worth more than they save.
    # Publish only a successful conversion; a failed partial must not be cached.
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".packing-") as staging:
        packed = Path(staging) / dest.name
        _run(
            exe,
            ["to-safetensors", str(checkpoint_dir), str(packed), "--verify"],
            "packing the hand-off",
        )
        if not packed.is_file():
            raise AuditError(f"{ENTRYPOINT} reported success but wrote no {dest.name}.")
        packed.replace(dest)
    return dest


def unpack(venv_dir: Path, bundle: Path, dest: Path) -> Path:
    """``handoff.safetensors`` -> a ``step_N/`` the replay can load."""
    exe = _require(venv_dir, "start from")
    if not bundle.is_file():
        raise AuditError(f"no hand-off bundle at {bundle}.")
    if dest.resolve() == bundle.resolve() or dest.resolve() in bundle.resolve().parents:
        raise AuditError("the unpack destination must not contain the source bundle.")
    # The converter refuses a destination that already holds files, so a
    # half-unpacked directory from an interrupted run is cleared rather than
    # merged into.
    if dest.exists() and any(dest.iterdir()):
        import shutil

        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    _run(exe, ["from-safetensors", str(bundle), str(dest), "--verify"], "unpacking the hand-off")
    if not (dest / "meta.json").is_file():
        raise AuditError(
            f"the unpacked hand-off at {dest} has no meta.json.",
            hint="A hand-off must carry the resume sidecar. One packed from a "
            "bare dcp/ directory or with --keys is weight-only transport, "
            "and a replay started from it diverges.",
        )
    return dest
