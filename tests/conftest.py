"""Fixtures built from a real kit's shape, not an invented one."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PRETRAIN_SHA = "3f60ef8ff6afbbbfef9fb2a16c6d9efb928fb32e"
REPOP_SHA = "244c0791e378a180dd6b29bbf8f2e244b2bea806"
INIT_HASH = "e8d5dce15fafd4e199485919aaf1286b93953ec0872f2d20fdfb0878739323e2"
REPOP_WHEEL = f"repop-0.1.5+g{REPOP_SHA[:12]}-cp311-cp311-macosx_14_0_arm64.whl"
PRETRAIN_WHEEL = "pretrain-0.1.0-py3-none-any.whl"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def kit_dir(tmp_path: Path) -> Path:
    """A kit laid out exactly as publish_audit_kit.sh publishes one."""
    d = tmp_path / "kit"
    d.mkdir()
    payloads = {REPOP_WHEEL: b"repop wheel bytes", PRETRAIN_WHEEL: b"pretrain wheel bytes"}
    for name, blob in payloads.items():
        (d / name).write_bytes(blob)

    trajectory = {
        "trajectory_format": 1,
        "name": "init-units-v1",
        "repop_commit": REPOP_SHA,
        "pretrain_commit": PRETRAIN_SHA,
        "notes": "config-only init units",
        "units": [
            {
                "kind": "init",
                "config_name": "1b_repop_run3",
                "until_step": 0,
                "state_hash": INIT_HASH,
                "devices_verified": ["cpu", "mps"],
                "mps_wall_seconds": 29.7,
                "mps_peak_rss_gb": 5.8,
            }
        ],
    }
    traj_bytes = (json.dumps(trajectory, indent=2) + "\n").encode()
    (d / "trajectory.json").write_bytes(traj_bytes)

    (d / "kit.json").write_text(
        json.dumps(
            {
                "kit_format": 1,
                "pretrain_commit": PRETRAIN_SHA,
                "repop_commit": REPOP_SHA,
                "files": [
                    {
                        "name": PRETRAIN_WHEEL,
                        "sha256": _sha(payloads[PRETRAIN_WHEEL]),
                        "bytes": len(payloads[PRETRAIN_WHEEL]),
                    },
                    {
                        "name": REPOP_WHEEL,
                        "sha256": _sha(payloads[REPOP_WHEEL]),
                        "bytes": len(payloads[REPOP_WHEEL]),
                    },
                    {
                        "name": "trajectory.json",
                        "sha256": _sha(traj_bytes),
                        "bytes": len(traj_bytes),
                    },
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return d


def write_handoff(
    d: Path,
    *,
    world: int = 1,
    safetensors: int | None = None,
    step: int = 200,
    omit: tuple[str, ...] = (),
) -> Path:
    """A COMPLETE hand-off, in the shape `audit_replay --save-checkpoint-dir`
    writes one: the DCP checkpoint, the per-rank batch-hasher chains, and the
    gradients sidecar the recipient needs to reconstruct the published hash.

    `omit` drops members, for the tests that check an incomplete hand-off is
    refused rather than sent. `safetensors` adds the single-file bundle the
    record's intake takes today, which the converter does not yet produce.
    """
    d.mkdir(parents=True, exist_ok=True)
    (d / "dcp").mkdir(exist_ok=True)
    (d / "dcp" / "__0_0.distcp").write_bytes(b"shard")
    files = {
        "meta.json": json.dumps({"step": step, "dp_world_size": world, "chained_hash": "a" * 64}),
        "_COMPLETE": "",
        "state_hash.txt": "a" * 64 + "\n",
        "global_stream.json": "{}",
        "rng.rank_0.pt": "rng",
        "gradients.safetensors": "grads",
    }
    for r in range(world):
        files[f"batch_hasher.rank_{r}.bin"] = "chain" + str(r)
    for name, body in files.items():
        if name not in omit:
            (d / name).write_text(body)
    if safetensors is not None:
        (d / "handoff.safetensors").write_bytes(b"x" * safetensors)
        (d / "handoff.json").write_text('{"format": "safetensors"}')
    return d


# The stubbed interval world is the fixture every end-to-end test stands in, so
# it is registered here rather than re-imported into each module that needs it.
import test_interval_end_to_end as _e2e

world = _e2e.world
