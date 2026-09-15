#!/usr/bin/env python3
"""Build a demo kit whose replay is stubbed, so the UI can be exercised.

A real interval audit is ~18 GB of checkpoint and most of a day, which makes
the micro-batch display the hardest part of this tool to look at. This assembles
a kit, a pre-provisioned venv and a mock record so `gensyn-audit run` drives its
entire production path -- provenance chain, step context, predecessor digest
gate, the live bar, the loss gate, submission, upload -- against a replay that
finishes in a minute.

The stub is deliberately faithful where it matters: it drives a real ``tqdm``
bar with audit_replay's own construction, so what the display parses is what
tqdm actually writes, not an imitation of it.

    python tools/make_demo_kit.py --dest ~/audit-demo --seconds 90

Not shipped in the wheel. Nothing it produces is an audit of anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

REPOP_SHA = "244c0791e378a180dd6b29bbf8f2e244b2bea806"
PRETRAIN_SHA = "3f60ef8ff6afbbbfef9fb2a16c6d9efb928fb32e"
RUN = "20260703-171943-4e85cd3"

# Real values, read from the published run. The digests are genuine; only the
# replay that would produce them is stubbed.
COMMITTED = "49a35d246590678b78088163da5c4600591e203d72beae008419ec5e479bdc2a"
PRED_DIGEST = "16b3d656df3b2c9ac9dd3dacdfad3a0132f7e17c0c175c0f369eae519d776e30"
# Synthetic — never the run's real withheld values.
TRUE_CE, TRUE_ZL = 4.8151623042000000, 0.0234567890123456
MICROBATCHES = 288

STUB = '''#!{python}
"""Stands in for pretrain-audit-replay. Emits what it emits, at demo speed."""
import argparse, json, os, sys, time
from pathlib import Path
from tqdm import tqdm

p = argparse.ArgumentParser()
for f in ("--checkpoint", "--device", "--until-step", "--expect-hash", "--gcs-root",
          "--fetch-dest", "--loss-log", "--save-checkpoint-dir", "--data-root",
          "--optimizer-offload-dir", "--master-offload-dir", "--config-name",
          "--descriptor-checkpoint"):
    p.add_argument(f)
p.add_argument("--from-init", action="store_true")
p.add_argument("--offload-optimizer", action="store_true")
p.add_argument("--offload-master", action="store_true")
a, _ = p.parse_known_args()

TOTAL = {seconds}
MB = {microbatches}
STEP = int(a.until_step or 200)

def log(level, msg):
    print(f"2026-09-07 10:00:00,000 {{level}} pretrain.audit :: {{msg}}", flush=True)

log("INFO", "repop_env: REPOP_EXECUTION_MODE=cross_device_reproducible")
log("INFO", f"auditing {{a.checkpoint}}: N=48 (dp_replicate=6 dp_shard=8) seed=42 "
            f"start_step={{STEP-100}} consumed_tokens=419430400")
log("INFO", "loss: fused CE+z-loss (matches the training loop)")
time.sleep(TOTAL * 0.06)
log("INFO", "MEMLOG[start] host_RSS=8.2GB mps_alloc=3.1GB mps_driver=7.4GB")
log("INFO", f"optimizer-state offload ON: 387 params' AdamW moments spilled to disk")
log("INFO", f"replaying steps {{STEP-100}} \\u2192 {{STEP}}")

# audit_replay's own construction (cli/audit_replay.py:1506), so the bytes the
# display parses are real tqdm output.
mb = tqdm(total=None, desc="  microbatches", unit="mb", dynamic_ncols=True,
          position=1, leave=False)
mb.reset(total=MB)
mb.set_description(f"  step {{STEP}} microbatches")
per = TOTAL * 0.88 / MB
for i in range(MB):
    time.sleep(per)
    mb.update(1)
    if i == MB // 2:
        log("INFO", "MEMLOG[mid-step] host_RSS=21.4GB mps_alloc=12.0GB mps_driver=19.8GB")
mb.close()

log("INFO", "MEMLOG[post-step] host_RSS=22.1GB mps_alloc=12.4GB mps_driver=20.3GB")
produced = a.expect_hash
losses = [{{"step": STEP, "consumed_tokens": 419840000,
           "loss_ce": {ce!r}, "loss_zloss": {zl!r}}}]
if a.loss_log:
    Path(a.loss_log).write_text(json.dumps({{"records": losses}}, indent=2) + "\\n")

saved = None
if a.save_checkpoint_dir:
    d = Path(a.save_checkpoint_dir) / f"step_{{STEP:09d}}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state_hash.txt").write_text(produced + "\\n")
    (d / "weights.bin").write_bytes(os.urandom(400_000))
    saved = str(d)
    log("INFO", f"saved chained-audit checkpoint \\u2192 {{saved}} (step={{STEP}} "
                f"consumed_tokens=419840000 chained_hash={{produced[:16]}})")

log("INFO", f"state_hash={{produced[:16]}} expected={{a.expect_hash[:16]}} MATCH=True")
res = {{"step": STEP, "consumed_tokens": 419840000, "state_hash": produced,
       "repop": {{"commit": "{repop}", "backends": ["cpu", "metal"],
                 "cuda_arch_list": ""}},
       "device": a.device or "mps", "rank0_losses": losses,
       "expected": a.expect_hash, "match": True}}
if saved:
    res["saved_checkpoint"] = saved
print(json.dumps(res, indent=2))
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default="~/audit-demo", help="where to build it")
    ap.add_argument(
        "--seconds",
        type=float,
        default=90.0,
        help="how long the stubbed replay should take (default 90)",
    )
    ap.add_argument(
        "--tqdm-python", help="an interpreter with tqdm (default: the kit venv you already have)"
    )
    args = ap.parse_args()

    dest = Path(args.dest).expanduser().resolve()
    if dest.exists():
        shutil.rmtree(dest)

    python = args.tqdm_python or _find_tqdm_python()
    if python is None:
        print(
            "No interpreter with tqdm found. Run a real init audit first so a kit "
            "venv exists, or pass --tqdm-python.",
            file=sys.stderr,
        )
        return 1

    kit = dest / "kit"
    kit.mkdir(parents=True)
    wheels = {
        f"repop-0.1.5+g{REPOP_SHA[:12]}-cp311-cp311-macosx_14_0_arm64.whl": b"demo repop wheel",
        "pretrain-0.1.0-py3-none-any.whl": b"demo pretrain wheel",
    }
    for name, blob in wheels.items():
        (kit / name).write_bytes(blob)

    traj = {
        "trajectory_format": 1,
        "name": "demo-units",
        "repop_commit": REPOP_SHA,
        "pretrain_commit": PRETRAIN_SHA,
        "notes": "DEMO ONLY. The replay behind these units is stubbed.",
        "units": [
            {
                "kind": "init",
                "config_name": "demo_init",
                "until_step": 0,
                "state_hash": "d" * 64,
                "devices_verified": ["mps"],
                "mps_wall_seconds": 10.0,
                "mps_peak_rss_gb": 1.0,
            }
        ],
    }
    tb = (json.dumps(traj, indent=2) + "\n").encode()
    (kit / "trajectory.json").write_bytes(tb)

    files = [
        {"name": n, "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
        for n, b in wheels.items()
    ]
    files.append(
        {"name": "trajectory.json", "sha256": hashlib.sha256(tb).hexdigest(), "bytes": len(tb)}
    )
    (kit / "kit.json").write_text(
        json.dumps(
            {
                "kit_format": 1,
                "pretrain_commit": PRETRAIN_SHA,
                "repop_commit": REPOP_SHA,
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )

    kit_id = f"pt-{PRETRAIN_SHA[:12]}_rp-{REPOP_SHA[:12]}"
    venv_bin = dest / "cache" / "kits" / kit_id / "venv" / "bin"
    venv_bin.mkdir(parents=True)

    build_info = json.dumps(
        {"commit": REPOP_SHA, "backends": ["cpu", "metal"], "cuda_arch_list": ""}
    )
    (venv_bin / "python").write_text(
        f'#!/bin/sh\nif [ "$1" = "-c" ]; then\n'
        f'  case "$2" in\n'
        f"    *build_info*) echo '{build_info}' ;;\n"
        f"    *python_version*) echo 3.11.6 ;;\n"
        f"  esac\nfi\n"
    )
    _exe(venv_bin / "python")

    (venv_bin / "pretrain-audit-replay").write_text(
        STUB.format(
            python=python,
            seconds=args.seconds,
            microbatches=MICROBATCHES,
            ce=TRUE_CE,
            zl=TRUE_ZL,
            repop=REPOP_SHA,
        )
    )
    _exe(venv_bin / "pretrain-audit-replay")

    pred = dest / "predecessor" / "step_000000100"
    pred.mkdir(parents=True)
    (pred / "state_hash.txt").write_text(PRED_DIGEST + "\n")
    (pred / "meta.json").write_text("{}")

    fixture = dest / "record.json"
    fixture.write_text(
        json.dumps(
            {
                "_comment": "DEMO fixture for the mock record. Digests are real; the "
                "replay that would produce them is stubbed.",
                "run": RUN,
                "gcs_root": None,
                "steps": {
                    "200": {
                        "committed_hash": COMMITTED,
                        "predecessor": {
                            "source": "published",
                            "step": 100,
                            "uri": str(pred),
                            "digest": PRED_DIGEST,
                        },
                        "phase": "warmup",
                        "microbatches": MICROBATCHES,
                    }
                },
                "withheld_losses": {"200": {"loss_ce": TRUE_CE, "loss_zloss": TRUE_ZL}},
            },
            indent=2,
        )
        + "\n"
    )

    runs = dest / "runs"
    runs.mkdir()
    print(f"Demo kit built at {dest}\n")
    print(
        f"Run this to watch the micro-batch UI (~{args.seconds:.0f}s, real tqdm, real code path):\n"
    )
    print(f"  cd {runs} && \\")
    print(f"  AUDIT_CACHE={dest / 'cache'} \\")
    print(f"  gensyn-audit run --kit {kit} \\")
    print(f"      --step 200 --run {RUN} \\")
    print(f"      --record mock://{fixture} \\")
    print("      --claim clm_demo --machine 'M4 Pro · 24 GB'\n")
    print(
        "Nothing here audits anything: the replay is a stub. It exists to make "
        "the display\nlookable-at without spending 18 GB and half a day."
    )
    return 0


def _exe(p: Path) -> None:
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _find_tqdm_python() -> str | None:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gensyn-audit"
    for py in sorted(base.glob("kits/*/venv/bin/python")):
        if (py.parent.parent / "lib").is_dir():
            return str(py)
    return None


if __name__ == "__main__":
    sys.exit(main())
