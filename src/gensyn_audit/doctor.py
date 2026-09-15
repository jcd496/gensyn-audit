"""Preflight, sized to the unit you actually asked for.

The old preflight assumed every audit was a half-day 1.6B step and demanded
48GB and 60GB of disk for all of them. With the kit's trajectory that is wrong
and unhelpful: an init unit needs about 6GB of RAM and under a minute, and
telling a volunteer with a 16GB Mac that they cannot participate would be
false.

So the checks read the unit's own published figures. Everything here reports
and prescribes; nothing installs, and nothing is repaired behind your back.
"""

from __future__ import annotations

import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import kit as kitmod
from .errors import AuditError
from .kit import Kit, Unit, build_info
from .plan import OPEN_FILES, UNSET_VARS, Plan

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"

#: Headroom over a unit's measured peak. Below this the machine is swapping,
#: which is correct but can turn hours into days.
_MEMORY_HEADROOM = 1.35

#: An interval replay's measured working set, when the trajectory does not say.
_INTERVAL_MEMORY_GB = 24.0
#: Unified-memory threshold below which interval replays are likely to swap.
#: Advisory only because lower-memory machines can still complete an audit.
_INTERVAL_COMFORT_GB = 40.0
_INTERVAL_TIMING = (
    "an audit step takes roughly 18 hours on a 24 GB MacBook Pro, "
    "6 hours on a 48 GB one, and under an hour on an H100."
)
#: venv + wheels + torch.
_KIT_DISK_GB = 6.0
#: Predecessor checkpoint, shards, handoff, offload spill -- and, on both ends
#: of the relay, gradients.safetensors: one tensor per parameter at the
#: parameter's own dtype, so ~1x the model's bf16 weights (~4 GB at 1.6B) is
#: downloaded with every crowd predecessor and uploaded with every hand-off.
_INTERVAL_DISK_GB = 68.0

#: The genesis step downloads no predecessor (the start state is regenerated),
#: so the ~19-26 GB an ordinary interval spends on one is not needed. What
#: remains is the shards the first step consumes, the hand-off it writes, and
#: the offload spill it cannot use -- see `_GENESIS_MEMORY_GB`.
_GENESIS_DISK_GB = 45.0

#: Provisional screening threshold, NOT a measured sufficient capacity.
#: From-init refuses optimizer offload, so the master, moments and backward
#: transients need more headroom. MPS/CPU check host capacity; CUDA checks free
#: VRAM on the selected device as well as host capacity for packing. These
#: different pools need independent full-size validation before release.
_GENESIS_MEMORY_GB = 48.0


@dataclass
class Check:
    name: str
    status: str
    value: str
    note: str = ""
    fix: str = ""

    @property
    def blocking(self) -> bool:
        return self.status == FAIL


def _run(argv: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


def _memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError):
        return None


def _needed_memory_gb(unit: Unit) -> tuple[float, str]:
    """What this unit actually costs, preferring the trajectory's own figure."""
    if unit.mps_peak_rss_gb:
        return unit.mps_peak_rss_gb * _MEMORY_HEADROOM, f"measured peak {unit.mps_peak_rss_gb} GB"
    if unit.is_init:
        return 8.0, "typical init unit"
    if unit.is_genesis:
        return _GENESIS_MEMORY_GB, "from-init replay; the optimizer cannot be offloaded"
    return _INTERVAL_MEMORY_GB, "interval replay with both offload flags"


def _check_platform(device: str) -> Check:
    machine, system = platform.machine(), platform.system()
    if device != "mps":
        return Check("device", SKIP, f"--device {device}", "Apple-Silicon checks skipped")
    if system != "Darwin" or machine != "arm64":
        return Check(
            "Apple Silicon",
            FAIL,
            f"{system}/{machine}",
            fix="The Metal path needs an Apple-Silicon Mac. --device cpu "
            "verifies the same digests on CPU kernels.",
        )
    proc = _run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=10)
    return Check("Apple Silicon", PASS, proc.stdout.strip() if proc.returncode == 0 else "arm64")


def _cuda_free_gb(venv: Path | None) -> float | None:
    if venv is None or not kitmod.venv_python(venv).is_file():
        return None
    try:
        proc = _run(
            [
                str(kitmod.venv_python(venv)),
                "-c",
                "import torch; print(torch.cuda.mem_get_info()[0] / 1024**3)",
            ],
            timeout=30,
        )
        value = float(proc.stdout.strip()) if proc.returncode == 0 else float("nan")
        return value if math.isfinite(value) and value >= 0 else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _check_genesis_memory(device: str, venv: Path | None) -> Check:
    need = _GENESIS_MEMORY_GB
    host = _memory_gb()
    note = (
        "from-init cannot offload the optimizer; provisional threshold, "
        "not a measured capacity requirement or a guarantee this replay fits"
    )
    if host is None or host < need:
        return Check(
            "memory",
            FAIL,
            "unknown" if host is None else f"{host:.0f} GiB host",
            note=note,
            fix=f"This path requires at least {need:.0f} GiB host memory at preflight. "
            "Use a larger host or audit a later step.",
        )
    if device == "cuda":
        free = _cuda_free_gb(venv)
        if free is None:
            return Check(
                "memory",
                FAIL,
                "CUDA memory unknown",
                note=note,
                fix="Could not measure free VRAM with the kit's Python. Provision the kit "
                "and check that CUDA is available, then rerun doctor. "
                "The probe respects CUDA_VISIBLE_DEVICES.",
            )
        if free < need:
            return Check(
                "memory",
                FAIL,
                f"{free:.1f} GiB free VRAM",
                note=note,
                fix=f"This path requires at least {need:.0f} GiB free VRAM on the selected "
                "GPU, separately from host RAM. Free GPU memory or audit a later step.",
            )
        value = f"{free:.1f} GiB free VRAM; {host:.0f} GiB host"
    else:
        value = f"{host:.0f} GiB {'unified' if device == 'mps' else 'host'} memory"
    return Check(
        "memory",
        WARN,
        value,
        note=note,
        fix="Full-size step-0 replay, handoff and peak-memory validation is pending.",
    )


def _check_memory(unit: Unit, device: str, venv: Path | None = None) -> Check:
    if unit.is_genesis:
        return _check_genesis_memory(device, venv)
    if device != "mps":
        return Check("memory", SKIP, "n/a")
    have = _memory_gb()
    if have is None:
        return Check("memory", WARN, "unknown", "could not read physical memory")
    need, why = _needed_memory_gb(unit)
    if have < need:
        return Check(
            "memory",
            WARN if unit.is_init else FAIL,
            f"{have:.0f} GB",
            note=f"this unit wants ~{need:.0f} GB ({why})",
            fix="Below this the machine swaps. An init unit will still finish; an "
            "interval replay can go from hours to days.",
        )
    if not unit.is_init and have < _INTERVAL_COMFORT_GB:
        return Check(
            "memory",
            WARN,
            f"{have:.0f} GB",
            note=f"works, but swaps heavily below ~{_INTERVAL_COMFORT_GB:.0f} GB",
            fix=f"For scale: {_INTERVAL_TIMING}\n"
            "It still works; expect the long form. On 24 GB machines this replay has "
            "also\nproduced hashes that matched nothing, differently each time, for "
            "reasons that were\nnot established. Before reading a NO MATCH from this "
            "machine as a finding about the\nrun, reproduce it on a machine with more "
            "memory. See docs/long-running-audits.md.",
        )
    return Check("memory", PASS, f"{have:.0f} GB", note=f"unit needs ~{need:.0f} GB")


def _already_on_disk_gb(workdir: Path) -> float:
    """Bytes a resumed audit will not have to fetch again.

    The checkpoint and the shards are the bulk of an interval's footprint, and
    a resumed run reuses both -- `download_prefix` skips every object already
    present at the right size. Counting them as still-needed is what turned a
    stopped audit into an unstartable one: the space they occupy was subtracted
    from free disk and then demanded again.
    """
    # Allocated blocks, not apparent size: fetch_interval places ~1700 SPARSE
    # placeholder .bin files for untouched shards so the loader can open all
    # readers. They report terabytes and occupy nothing, and counting st_size
    # here claimed 1701 GB was already fetched.
    # `verified-predecessor` is the unpacked hand-off; the bundle it came from
    # is deleted once the gate passes, so on a restart this directory is the
    # predecessor's whole footprint and the only thing standing in for it.
    total = 0
    for name in ("checkpoint", "verified-predecessor", "data"):
        d = workdir / name
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file():
                    st = f.stat()
                    # `blocks * 512 or size` would be exactly wrong: a fully
                    # sparse file has st_blocks == 0, so the fallback fires on
                    # the one case this exists to handle.
                    total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
    return total / 1024**3


def _check_disk(workdir: Path, unit: Unit) -> Check:
    probe = workdir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / 1024**3
    except OSError as exc:
        return Check("free disk", WARN, "unknown", str(exc))

    if unit.is_init:
        need = _KIT_DISK_GB
    elif unit.is_genesis:
        need = _KIT_DISK_GB + _GENESIS_DISK_GB
    else:
        need = _KIT_DISK_GB + _INTERVAL_DISK_GB
    have_gb = 0.0 if unit.is_init else _already_on_disk_gb(workdir)
    need = max(need - have_gb, _KIT_DISK_GB)
    resumed = f", {have_gb:.0f} GB already fetched" if have_gb >= 1 else ""

    if free_gb < need:
        return Check(
            "free disk",
            FAIL,
            f"{free_gb:.0f} GB free",
            note=f"need ~{need:.0f} GB{resumed}",
            fix="The kit's venv alone is ~6 GB (torch). An interval replay adds "
            "the checkpoint, the shards, the handoff and the offload spill.",
        )
    return Check("free disk", PASS, f"{free_gb:.0f} GB free", note=f"need ~{need:.0f} GB{resumed}")


def _check_python() -> Check:
    """The published repop wheels are cp311. This is an ABI fact, not taste.

    What counts is an interpreter the kit venv can be built from, which is not
    the same question as what is on PATH: `uv tool install --python 3.11`
    leaves nothing called python3.11 on PATH and runs the tool on 3.11 anyway.
    Asking PATH alone failed the supported install and sent the reader off to
    install Python by hand (2026-09-11).
    """
    found = kitmod.find_base_interpreter()
    if not found:
        return Check(
            "python 3.11",
            FAIL,
            "none available",
            fix="The repop audit wheels are built for CPython 3.11 and will not "
            "install on another minor version. Let uv bring its own:\n"
            "  uv tool install --python 3.11 gensyn-audit",
        )
    proc = _run([found, "-V"], timeout=15)
    ours = found == sys.executable
    return Check(
        "python 3.11",
        PASS,
        proc.stdout.strip() or found,
        note="this tool's own interpreter" if ours else found,
    )


def _check_open_files(unit: Unit) -> Check:
    if unit.is_init:
        return Check("open files", SKIP, "init unit reads no shards")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= OPEN_FILES:
        return Check("open files", PASS, str(soft))
    if hard == resource.RLIM_INFINITY or hard >= OPEN_FILES:
        return Check(
            "open files",
            PASS,
            f"{soft} (raised at launch)",
            note=f"`gensyn-audit run` sets {OPEN_FILES}",
        )
    return Check(
        "open files",
        FAIL,
        f"soft={soft} hard={hard}",
        note=f"need {OPEN_FILES}",
        fix=f"sudo launchctl limit maxfiles {OPEN_FILES} {OPEN_FILES}\n"
        "then start a new shell. ~1700 shard memmaps otherwise fail with "
        '"too many open files".',
    )


def _check_stray_env() -> Check:
    stray = [v for v in UNSET_VARS if v in os.environ]
    if not stray:
        return Check("environment", PASS, "clean")
    return Check(
        "environment",
        WARN,
        ", ".join(f"{v} set" for v in stray),
        note="cleared for the replay",
        fix="`gensyn-audit run` removes these from the child environment. They point "
        "repop and pretrain at directories other than the ones inside the "
        "installed wheels, which is exactly what the kit exists to pin.",
    )


def _check_kit(kit: Kit, venv: Path, device: str) -> list[Check]:
    checks = [Check("kit", PASS, kit.kit_id, note=kit.public_prefix)]
    python = venv / "bin" / "python"
    if not python.is_file():
        return checks + [
            Check(
                "audit venv",
                WARN,
                "not provisioned yet",
                note="`gensyn-audit run` installs it",
                fix=f"Two wheels into {venv}. About 6 GB, once per kit.",
            )
        ]
    try:
        info = build_info(python)
    except AuditError as exc:
        return checks + [
            Check(
                "audit venv", FAIL, "repop not importable", note=str(exc)[:70], fix=exc.hint or ""
            )
        ]

    if info.get("commit") != kit.repop_commit:
        checks.append(
            Check(
                "repop build",
                FAIL,
                (info.get("commit") or "?")[:12],
                note=f"kit pins {kit.repop_commit[:12]}",
                fix="A result naming a different kernel build proves nothing about the "
                f"published trajectory. Delete {venv} and re-run to reinstall.",
            )
        )
    else:
        checks.append(Check("repop build", PASS, info["commit"][:12], note="matches kit.json"))

    needed = {"mps": "metal", "cuda": "cuda", "cpu": "cpu"}[device]
    backends = info.get("backends") or []
    if needed in backends:
        checks.append(Check("repop backends", PASS, ", ".join(backends)))
    else:
        checks.append(
            Check(
                "repop backends",
                FAIL,
                ", ".join(backends) or "none",
                note=f"--device {device} needs {needed!r}",
                fix="A build without it falls back to CPU kernels op by op "
                "and would 'pass' without touching the hardware the "
                "verification claims to cover.",
            )
        )
    return checks


#: What audit_replay demands of a checkpoint's descriptor before it will
#: replay it (cli/audit_replay.py, three separate ValueErrors). Each is a
#: property of the RUN that produced the checkpoint, recorded in its meta.json.
#: Directory checkpoints expose it before the large download; packed handoffs
#: expose it after their digest has been verified and they are unpacked.
_DESCRIPTOR_GUARDS = {
    "reduction_mode": (
        "deterministic_allgather",
        (
            "the run was not trained in auditable mode, so it cannot be reproduced "
            "bitwise on one device at all."
        ),
    ),
    "replicate_reduce_algo": (
        "recursive_doubling",
        ("a pre-reset checkpoint; the cross-replica fold cannot be replayed from this tree."),
    ),
    "clip_algo": (
        "global",
        (
            "this checkpoint predates the stateless deterministic global-norm clip. "
            "Audit it from an older checkout, or pick a step after the run's clipper "
            "boundary."
        ),
    ),
}


def _check_descriptor(unit: Unit) -> list[Check]:
    """Read the predecessor's meta.json and apply audit_replay's own guards.

    For directory checkpoints this is fetched before anything large.
    audit_replay applies the same checks after loading either checkpoint form.
    """
    if unit.is_init or not unit.checkpoint_uri:
        return []
    if unit.is_genesis:
        # Same file, different role: for a from-init replay this checkpoint is
        # the run DESCRIPTOR. The guards below are exactly what audit_replay
        # requires of it, so they are worth running -- and cost one 5 KB read.
        pass
    if unit.checkpoint_uri.rstrip("/").endswith(".safetensors"):
        return [
            Check(
                "checkpoint descriptor",
                SKIP,
                "deferred until authenticated unpack",
                note="packed handoff embeds meta.json; replay validates it before training",
            )
        ]

    from . import gcs

    uri = unit.checkpoint_uri.rstrip("/") + "/meta.json"
    try:
        meta = json.loads(gcs.get(uri).decode())
    except AuditError as exc:
        cheap = (
            "Checked before the download so a mismatch is cheap. The replay "
            "re-checks it either way.\n" + (exc.hint or "")
        )
        return [
            Check(
                "checkpoint descriptor",
                WARN,
                "could not read meta.json",
                note=str(exc)[:70],
                fix=cheap,
            )
        ]
    except (ValueError, UnicodeDecodeError) as exc:
        return [Check("checkpoint descriptor", WARN, "meta.json is unreadable", note=str(exc)[:70])]

    out = []
    for key, (want, why) in _DESCRIPTOR_GUARDS.items():
        got = meta.get(key, want)  # absent means the default, which is the wanted one
        if got != want:
            out.append(
                Check(
                    f"descriptor: {key}",
                    FAIL,
                    str(got),
                    note=f"audit_replay requires {want!r}",
                    fix=why,
                )
            )
    if out:
        return out
    return [
        Check(
            "checkpoint descriptor",
            PASS,
            f"step {meta.get('step')}",
            note="clip, reduction and fold all auditable",
        )
    ]


def _check_credentials(unit: Unit) -> list[Check]:
    """Can we read the artifacts? No Cloud SDK involved.

    Downloads go over plain HTTPS, so the only question is whether a token is
    needed at all. Public artifacts need none -- which is the point, and the
    intended end state for an audit anyone can run.
    """
    if unit.is_init or not unit.gcs_root:
        return [Check("artifact access", SKIP, "this unit downloads nothing")]

    from . import gcs

    try:
        creds = gcs.credentials()
    except AuditError as exc:
        return [
            Check(
                "artifact access",
                FAIL,
                "credentials unusable",
                note=str(exc)[:70],
                fix=exc.hint or "",
            )
        ]

    from .gcs import crc32c_implementation

    impl = crc32c_implementation()
    speed = Check(
        "checksum backend", PASS, impl, note="verifies at C speed" if impl != "python" else ""
    )
    if impl == "python":
        speed = Check(
            "checksum backend",
            WARN,
            "pure python",
            note="~20 MB/s — the download will be checksum-bound",
            fix="Checkpoint shards are composite objects with no md5, so crc32c "
            "verifies the whole ~19 GB. The C extension does it 300x faster:\n"
            "  pip install google-crc32c",
        )

    if creds.anonymous:
        # Not a failure: it is the target state. Whether it works is decided by
        # the first read, which says exactly what is missing if it 403s.
        return [
            Check(
                "artifact access", PASS, "anonymous", note="public artifacts need no credentials"
            ),
            speed,
        ]

    try:
        creds.token()
    except AuditError as exc:
        return [
            Check(
                "artifact access",
                FAIL,
                "credentials expired",
                note=creds.source[:60],
                fix=exc.hint or "",
            )
        ]
    return [
        Check("artifact access", PASS, "authenticated", note=creds.source.split("(")[0].strip()),
        speed,
    ]


def _check_predecessor_digest(ctx) -> list[Check]:
    """Check record provenance and digest availability before downloading."""
    if ctx is None:
        return []
    if ctx.predecessor.is_initial_weights:
        # Nothing is downloaded for the start state, so there is no digest to
        # publish. What must exist is the descriptor and the init commitment;
        # the record is refused above if it names a from-init predecessor
        # without both.
        if ctx.predecessor.genesis is None:
            return [
                Check(
                    "predecessor",
                    FAIL,
                    "from-init without a descriptor",
                    fix="The record must name the run descriptor and the published "
                    "init commitment for the first step.",
                )
            ]
        return [
            Check(
                "predecessor",
                PASS,
                "run initialization",
                note="regenerated from the seed; the replay checks it against the "
                "published init hash",
            )
        ]
    if not (ctx.predecessor.is_trusted_anchor or ctx.predecessor.is_crowd_provided):
        return [
            Check(
                "predecessor digest",
                FAIL,
                "unknown provenance",
                fix="The record must identify the predecessor's provenance.",
            )
        ]
    if ctx.predecessor.is_crowd_provided and ctx.predecessor.digest:
        if not ctx.predecessor.has_bundle_digest:
            return [
                Check(
                    "predecessor digest",
                    FAIL,
                    "invalid bundle digest",
                    fix="predecessor.digest must be 64 hexadecimal characters.",
                )
            ]
        return [Check("predecessor digest", PASS, "bundle digest published (BLAKE2b-256)")]
    if ctx.predecessor.artifact_files:
        return [
            Check(
                "predecessor digest",
                PASS,
                f"{len(ctx.predecessor.artifact_files)} file(s) published",
            )
        ]
    if ctx.predecessor.is_trusted_anchor:
        return [Check("predecessor", PASS, "Trusted Gensyn anchor")]
    return [
        Check(
            "predecessor digest",
            FAIL,
            "none published",
            note="the record publishes no predecessor.digest or artifactFiles for this hand-off",
            fix="A crowd hand-off has no override: it is another auditor's bytes.",
        )
    ]


def run_checks(plan: Plan, *, deep: bool = True, ctx=None, for_replay: bool = True) -> list[Check]:
    """Everything, ordered host-first. `deep=False` skips the venv import.

    `for_replay=False` drops the headroom checks. `run` re-invoked on a workdir
    that already holds a finished replay reports it rather than replaying, and
    that path deliberately keeps the offload spill: sizing the disk for a
    replay nobody is about to start would fail the report on the very bytes it
    exists to preserve.
    """
    checks = [
        _check_platform(plan.device),
        _check_python(),
        _check_open_files(plan.unit),
        _check_stray_env(),
    ]
    if for_replay:
        checks[1:1] = [
            _check_memory(plan.unit, plan.device, plan.venv),
            _check_disk(plan.workdir.root, plan.unit),
        ]
    if deep:
        checks += _check_kit(plan.kit, plan.venv, plan.device)
    checks += _check_credentials(plan.unit)
    # Always, even shallow: it is 5 KB and it is the check most likely to save
    # someone a wasted download.
    checks += _check_descriptor(plan.unit)
    checks += _check_predecessor_digest(ctx)
    return checks
