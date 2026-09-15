"""Turning a trajectory unit into an executable replay.

Everything the auditor would otherwise retype lives here and nowhere else, so
``gensyn-audit plan`` and ``gensyn-audit run`` can never disagree: the entrypoint, the flags
for each unit kind, and the small set of environment variables the kit does not
already handle.

Most of what the old source-checkout runbook needed is simply gone. There is no
``PYTHONPATH`` -- the wheels put ``pretrain`` and ``repop`` on the venv's own
path, and the wheel ships ``configs/`` as ``pretrain._configs`` so
``--config-name`` resolves with no repo. What remains is the memory and
bookkeeping knobs, and one variable that must be *absent*.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .errors import AuditError
from .kit import Kit, Unit, local_uri_path, replay_entrypoint

#: `ulimit -n`. An interval memmaps ~1700 shards; macOS defaults to 256.
#: Init units touch no data and do not need it.
OPEN_FILES = 65536

#: Set for every replay.
_ENV_ALWAYS = {
    "WANDB_MODE": "disabled",
}

#: Added for an interval replay on MPS. Init units allocate once and exit, so
#: none of this applies to them.
_ENV_INTERVAL_MPS = {
    # Trim the allocator cache per micro-batch. Costs speed (a device sync and
    # a realloc each time) and buys ~84GB -> ~43GB of peak, which is the
    # difference between running and swapping on a 24-48GB Mac.
    "PRETRAIN_AUDIT_EMPTY_CACHE_PER_MB": "1",
    # Host RSS + MPS allocation per phase, so `gensyn-audit status` can tell the
    # auditor whether they are in swap.
    "PRETRAIN_AUDIT_MEMLOG": "1",
    # Metal-4 int8 GEMM fast path: ~2x on GEMM, byte-exact, cannot change the
    # reproduced hash.
    "REPOP_INT8_MPP": "1",
    # Not a memory lever despite the name: same peak, ~47% faster.
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
}

#: Must be absent, not empty. repop.ops defaults it to the shader directory
#: inside the installed package; any override breaks metallib resolution, and
#: an empty string is still an override.
UNSET_VARS = ("REPOP_METAL_SHADER_DIR", "PRETRAIN_CONFIGS")


def cache_root() -> Path:
    """Where kits and their venvs live.

    Deliberately outside the per-step workdir: the venv is ~2GB of torch, it is
    keyed by kit id, and every audit against the same kit should share it.
    """
    env = os.environ.get("AUDIT_CACHE")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "gensyn-audit"


@dataclass
class KitPaths:
    """Where one kit is staged and installed. Shared across audits."""

    kit_id: str

    @property
    def root(self) -> Path:
        return cache_root() / "kits" / self.kit_id

    @property
    def stage(self) -> Path:
        return self.root / "stage"

    @property
    def venv(self) -> Path:
        return self.root / "venv"


@dataclass
class Workdir:
    """Per-audit layout, so every command can find things by name."""

    root: Path

    @property
    def log(self) -> Path:
        return self.root / "audit.log"

    @property
    def state(self) -> Path:
        return self.root / "run.json"

    @property
    def cli_log(self) -> Path:
        """Where a detached `run` writes its own screen: the kit and preflight
        blocks, the verdict, the submission and the upload. `audit.log` is the
        replay's output and stays separate."""
        return self.root / "gensyn-audit.log"

    @property
    def predecessor_record(self) -> Path:
        """The gate's verdict about the starting checkpoint, kept so a report
        made later -- by another invocation, from a workdir whose predecessor
        has since been cleaned up -- still says what was established."""
        return self.root / "predecessor.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def loss_log(self) -> Path:
        return self.root / "losses.json"

    @property
    def checkpoint_root(self) -> Path:
        return self.root / "checkpoint"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def handoff(self) -> Path:
        return self.root / "handoff"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def verified_predecessor(self) -> Path:
        """Where a packed crowd hand-off is unpacked once its bundle has passed
        the gate. This directory, not the bundle, is what the replay reads."""
        return self.root / "verified-predecessor"

    def checkpoint_dir(self, step: int) -> Path:
        return self.checkpoint_root / f"step_{step:09d}"

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def clear_scratch(self) -> int:
        """Discard the offload spill, returning the bytes reclaimed.

        `--optimizer-offload-dir` and `--master-offload-dir` are regenerated
        from scratch by every replay, so a spill left behind by one that died
        is pure garbage — and at ~18 GB it is enough to fail the disk preflight
        on the retry, which is how a stopped audit becomes an unstartable one.
        """
        if not self.scratch.exists():
            return 0
        freed = sum(f.stat().st_size for f in self.scratch.rglob("*") if f.is_file())
        shutil.rmtree(self.scratch, ignore_errors=True)
        return freed

    def consume_predecessor_bundle(self, bundle: Path | None) -> int:
        """Delete a downloaded hand-off bundle the gate has finished with,
        returning the bytes reclaimed.

        Once `verify.gate` has digest-checked the bundle, unpacked it into
        `verified_predecessor` and reconstructed the tensor state from that
        directory, nothing reads the bundle again: the replay starts from the
        unpacked directory, and `--restart` replays from it too (see
        `predecessor.json`). Leaving it was ~18 GB with no consumer for the
        life of the audit.

        Only a bundle *this workdir downloaded* -- a regular file under its own
        `checkpoint/` -- is consumed. A bundle the gate found anywhere else is
        someone's: the previous workdir's `handoff/handoff.safetensors` that
        `handoffs.find` pointed at is both that audit's upload source and the
        next step's reuse source, and `--checkpoint` names a file the auditor
        holds for their own reasons. Those are left exactly where they are.
        """
        if bundle is None:
            return 0
        try:
            resolved = bundle.resolve(strict=True)
            # Resolve the workdir, not a checkpoint/ link to someone else's files.
            root = self.root.resolve() / "checkpoint"
        except OSError:
            return 0
        if bundle.is_symlink() or not resolved.is_file() or root not in resolved.parents:
            return 0
        size = resolved.stat().st_size
        try:
            resolved.unlink()
        except OSError:
            return 0
        return size


@dataclass
class Plan:
    """A fully-resolved replay: what to run, with what environment, where."""

    unit: Unit
    kit: Kit
    workdir: Workdir
    venv: Path
    device: str = "mps"
    save_handoff: bool = True
    checkpoint: Path | None = None
    run: str = ""
    audit_step: int | None = None
    """The record's number for this step. `unit.until_step` is the log number."""
    extra_args: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_init(self) -> bool:
        return self.unit.is_init

    @property
    def is_genesis(self) -> bool:
        """The run's first step: replayed from the regenerated initialization."""
        return self.unit.is_genesis

    @property
    def needs_data(self) -> bool:
        return not self.is_init

    @property
    def needs_open_files(self) -> bool:
        return not self.is_init

    # ── environment ──────────────────────────────────────────────────────────
    def env_overlay(self) -> dict[str, str]:
        """The variables we set.

        Everything repop needs to match the trained run -- REPOP_EXECUTION_MODE,
        the Hadamard flags, CUBLAS_WORKSPACE_CONFIG -- is applied by
        audit_replay from the checkpoint's own meta.json. Setting those by hand
        is how you audit a different run than the one you downloaded.
        """
        env = dict(_ENV_ALWAYS)
        if not self.is_init and self.device == "mps":
            env.update(_ENV_INTERVAL_MPS)
        return env

    # ── command ──────────────────────────────────────────────────────────────
    def argv(self) -> list[str]:
        args = [str(replay_entrypoint(self.venv))]

        if self.is_init:
            # No checkpoint, no data, no offload: the init state is regenerated
            # from the seed and is device-independent by construction.
            args += [
                "--from-init",
                "--until-step",
                "0",
                "--config-name",
                self.unit.config_name,
                "--device",
                self.device,
                "--expect-hash",
                self.unit.target_hash,
            ]
            return args + self.extra_args

        args += [
            "--checkpoint",
            str(self.checkpoint_path()),
            "--device",
            self.device,
            "--until-step",
            str(self.unit.until_step),
            "--expect-hash",
            self.unit.target_hash,
        ]
        if self.is_genesis:
            # The start state is rebuilt from the seed; the checkpoint above is
            # the run descriptor. audit_replay gates the regenerated state
            # against state_hash_init.txt (staged beside it) before replaying,
            # and refuses --descriptor-checkpoint on this path.
            args += ["--from-init"]
        if self.unit.descriptor_uri and not self.is_genesis:
            # The one interval that crosses a fork: state from --checkpoint,
            # but the segment-scoped descriptor (clipper, repop_env, topology,
            # resolved config) from the segment that actually trained it.
            # Without this the replay uses the old segment's rules and
            # mismatches on the first step.
            args += ["--descriptor-checkpoint", str(self.descriptor_path())]
        if self.unit.gcs_root:
            args += ["--gcs-root", self.unit.gcs_root, "--fetch-dest", str(self.workdir.data)]
        else:
            args += ["--data-root", str(self.workdir.data / "shards")]
        if self.device == "mps" and not self.is_genesis:
            # ~12GB of moments and ~6GB of fp32 master off the unified pool.
            # Both are bitwise-identical; they only move bytes to disk.
            #
            # Not on the genesis path: audit_replay refuses --offload-optimizer
            # with --from-init, so the optimizer state stays resident
            # and the machine needs the headroom instead. `doctor` says so
            # before anything starts rather than letting the replay raise.
            args += [
                "--offload-optimizer",
                "--offload-master",
                "--optimizer-offload-dir",
                str(self.workdir.scratch / "optimizer"),
                "--master-offload-dir",
                str(self.workdir.scratch / "master"),
            ]
        # The per-step losses the record's gate runs on, written atomically
        # after every step so they survive a killed process.
        args += ["--loss-log", str(self.workdir.loss_log)]
        if self.save_handoff:
            # The relay. audit_replay refuses to write it when the hash did not
            # match, so it is never a way to launder an unreproduced state.
            args += ["--save-checkpoint-dir", str(self.workdir.handoff)]
        return args + self.extra_args

    def descriptor_path(self) -> Path:
        """Where the fork's meta.json is staged. audit_replay reads only that
        one file from --descriptor-checkpoint, so nothing else is fetched."""
        return self.workdir.root / "descriptor"

    def genesis_root(self) -> Path:
        """Where the run descriptor is staged for a from-init replay.

        The layout is the published bucket's, not a convenience of ours:
        audit_replay looks for ``state_hash_init.txt`` in the PARENT of
        --checkpoint, exactly where the run publishes it beside the checkpoint
        directories. Flattening the two into one directory would silently skip
        the init comparison, which is the only thing standing between a
        from-init replay and starting from an unchecked state.
        """
        return self.workdir.root / "genesis"

    def genesis_descriptor_path(self) -> Path:
        descriptor, _ = self.genesis_sources()
        # The predecessor is initialization (step 0), but the metadata comes
        # from a later checkpoint. Match fetch_genesis, including URI overrides.
        return self.genesis_root() / Path(descriptor.rstrip("/")).name

    def genesis_sources(self) -> tuple[str, str]:
        """Where the descriptor and the init commitment are published.

        A genesis unit carries both by construction — `_unit_from_step` refuses
        a from-init predecessor the record has half-described — so this states
        the invariant rather than defending against a caller. If it ever fires,
        something built a from-init unit by another route and the replay would
        otherwise start from a state nothing compared.
        """
        descriptor, init_hash = self.unit.checkpoint_uri, self.unit.init_state_hash_uri
        if not descriptor or not init_hash:
            raise AuditError(
                "this from-init unit names no run descriptor or no init commitment.",
                hint="Both come from the record's `predecessor.genesis` block.",
            )
        return descriptor, init_hash

    def checkpoint_path(self) -> Path:
        if self.checkpoint is not None:
            return self.checkpoint
        if self.is_genesis:
            # --from-init reads this for its meta.json alone. Its tensors are
            # never loaded, so nothing large is staged here.
            return self.genesis_descriptor_path()
        # A predecessor already on this machine is used where it is, not copied
        # into the workdir first. Planning a workdir path for it would hand
        # audit_replay a directory nothing ever populates.
        if self.unit.checkpoint_uri:
            local = local_uri_path(self.unit.checkpoint_uri)
            if local is not None:
                return local.resolve()
        step = self.unit.predecessor_step
        if step is None:
            raise AuditError(
                "this unit does not say which step its predecessor checkpoint is.",
                hint="It comes from the step API's `predecessor.step`. Deriving it as "
                "until_step - 1 would name a checkpoint the run never saved: "
                "checkpoints are periodic, not per-step.",
            )
        return self.workdir.checkpoint_dir(step)

    # ── display ──────────────────────────────────────────────────────────────
    def shell_script(self) -> str:
        """The whole thing as a volunteer would type it, for `gensyn-audit plan`."""
        from .ui import render_command, shell_quote

        lines: list[str] = []
        if self.needs_open_files:
            lines.append(f"ulimit -n {OPEN_FILES}")
        for key, value in sorted(self.env_overlay().items()):
            lines.append(f"export {key}={shell_quote(value)}")
        for key in UNSET_VARS:
            lines.append(f"unset {key}")
        lines.append("")
        lines.append(render_command(self.argv(), indent="").lstrip())
        return "\n".join(lines)


def build(
    *,
    unit: Unit,
    kit: Kit,
    workdir: Path,
    venv: Path,
    device: str = "mps",
    save_handoff: bool = True,
    checkpoint: Path | None = None,
    run: str = "",
    audit_step: int | None = None,
    extra_args: list[str] | None = None,
) -> Plan:
    plan = Plan(
        unit=unit,
        kit=kit,
        workdir=Workdir(workdir.expanduser().resolve()),
        venv=venv,
        device=device,
        save_handoff=save_handoff and not unit.is_init,
        checkpoint=checkpoint.expanduser().resolve() if checkpoint else None,
        run=run or unit.config_name,
        audit_step=audit_step,
        extra_args=list(extra_args or []),
    )

    if unit.descriptor_uri:
        plan.notes.append(
            "This interval crosses a fork boundary. The starting state comes from "
            "the previous segment's checkpoint, but the interval was trained under "
            "the next segment's descriptor, so --descriptor-checkpoint overlays it. "
            "Without that the replay mismatches on the first step and looks like a "
            "divergence when it is not."
        )
    if unit.is_genesis:
        plan.notes.append(
            "This is the run's first step, so there is no checkpoint to start "
            "from: the initial weights are regenerated from the published seed "
            "and compared against the run's own init hash before the replay "
            "begins. Nothing but a few kilobytes of descriptor is downloaded "
            "for the start state, and the optimizer offload is unavailable on "
            "this path, so it needs more memory than a later step."
        )
    if unit.is_init:
        plan.notes.append(
            "An init unit downloads nothing and reads no checkpoint: it regenerates "
            "the run's seeded initial state and compares its digest. This is the "
            "whole verification, and it costs seconds."
        )
    if device not in unit.devices_verified and unit.devices_verified:
        plan.notes.append(
            f"The trajectory records this unit as verified on "
            f"{', '.join(unit.devices_verified)} -- not {device}. The digests are "
            "device-independent by design, so a match here is still a match; it is "
            "just the first one on this backend."
        )
    return plan
