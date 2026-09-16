"""Command line.

The shape the PRD asks for is one command: `gensyn-audit run` resolves the kit,
preflights, installs what it needs, replays, compares, and reports. It is
idempotent — a staged file with the right digest is not re-fetched and a venv
already carrying the kit's repop is not reinstalled — so an interrupted audit
resumes by running the same command again rather than a different one.

The other subcommands are windows onto that pipeline, not steps you have to
drive: `doctor` is its preflight, `plan` is the command it would run, `units`
is the trajectory it would run against.

Exit codes: 0 match, 1 the tool or the environment failed, 2 the replay
finished and the hash did not match.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from . import __version__, brand, doctor, fetch, runner
from . import commitments as commitmentsmod
from . import convert as convertmod
from . import handoffs as handoffs_mod
from . import kit as kitmod
from . import manifest as manifestmod
from . import mock as mockmod
from . import plan as plan_mod
from . import progress as progress_mod
from . import record as recordmod
from . import submit as submit_mod
from . import upload as uploadmod
from . import verify as verifymod
from .errors import AuditError
from .outcome import INCONCLUSIVE_EDUCATION, NO_MATCH_EDUCATION, Outcome, classify
from .steps import StepRef
from .ui import (
    ARROW,
    FAIL,
    PASS,
    SKIP,
    WARN,
    Activity,
    Live,
    bar,
    echo,
    head,
    human_bytes,
    human_duration,
    kv,
    paint,
    rule,
    set_plain,
    warn,
)

EXIT_OK, EXIT_ERROR, EXIT_MISMATCH = 0, 1, 2

# Deliberately no default bucket. A verification tool that assumes where a
# run's bytes live cannot verify a run that moved, and a wrong default is worse
# than a missing one: it fails late, with a permissions error, instead of
# saying which location it needed.

_PLATFORM_TAG = {"darwin": "macosx", "linux": "linux"}.get(sys.platform, sys.platform)
_DEFAULT_DEVICE = {"darwin": "mps", "linux": "cuda"}.get(sys.platform, "cpu")


# ── resolution ───────────────────────────────────────────────────────────────


def _resolve_kit(args: argparse.Namespace, manifest=None) -> tuple[kitmod.Kit, kitmod.Trajectory]:
    prefix = args.kit or (manifest.kit if manifest else None)
    if prefix is None:
        raise AuditError(
            "no kit given, and no manifest names one.",
            hint="A kit is one immutable prefix holding kit.json, the wheels and\n"
            "trajectory.json. Either:\n"
            "  gensyn-audit run --manifest <url>          # names the kit, record and roots\n"
            "  gensyn-audit run --kit <prefix>            # the kit directly",
        )
    k = kitmod.load_kit(prefix)
    return k, kitmod.load_trajectory(k)


def _resolve_record(args: argparse.Namespace, manifest=None) -> recordmod.Record:
    """The live API if one was named, else the in-process mock.

    Nothing downstream knows which it got — that is the point of the seam — so
    the announcement happens here, once, and loudly.
    """
    url = getattr(args, "record", None) or (manifest.record if manifest else None)
    if url and not url.startswith("mock://"):
        rec = recordmod.HttpRecord(url)
        echo(kv("record", rec.label))
        return rec
    fixture = Path(url[len("mock://") :]) if url and url != "mock://" else mockmod.default_fixture()
    if fixture is None:
        raise AuditError(
            "no record URL given.",
            hint="Pass a manifest URL from the audit website.",
        )
    rec = mockmod.MockRecord(fixture)
    echo(kv("record", "mock", str(fixture)))
    echo(mockmod.MockRecord.banner())
    return rec


def _warn_if_simulated(record, args: argparse.Namespace) -> None:
    """Refuse only what cannot succeed. Warn about the rest.

    There are two different things a record can admit to, and they deserve
    different answers:

    *Placeholder commitments* make the replay pointless. The reproduced hash is
    compared against one the run never produced, so it cannot match however
    correct the replay was. Twelve hours to learn nothing is the worst outcome
    this tool can deliver, so that is a refusal.

    *An unwired loss gate* does not invalidate anything. The runner's job is to
    reproduce the step and report the losses; deciding whether those losses are
    the cluster's is the verification service's job, and it is the only party
    that can -- it holds the withheld values and the runner never does. With the
    gate unwired the hash comparison is still real and the losses are still
    correctly produced; what is provisional is only the record's *acceptance*.
    That is a warning, not a wall.
    """
    if record is None or getattr(record, "is_mock", False):
        return
    try:
        rm = record.manifest()
    except AuditError:
        return  # a manifest we cannot read is not grounds to refuse
    if not rm.simulated:
        return

    echo()
    warn(f"This record reports itself as SIMULATED ({rm.run_name}).")
    for key, why in sorted(rm.simulated_parts.items()):
        # A part is listed with a null reason when it is genuine; printing
        # "commitments: None" reads as though it were simulated.
        if why:
            echo(paint(f"    {key}: {why}", "dim"))

    if args.step is None:
        return  # an init unit is verified locally; the record is not involved

    if not rm.commitments_are_placeholders:
        if not rm.loss_gate_is_live:
            echo(
                paint(
                    "    Your replay and its hash comparison are unaffected. What is "
                    "provisional\n    is the record's acceptance: nobody checks the "
                    "losses you report.",
                    "dim",
                )
            )
        return

    if getattr(args, "allow_simulated", False):
        echo(
            paint("    --allow-simulated given; continuing against placeholder commitments.", "dim")
        )
        return

    raise AuditError(
        "refusing to start an interval replay against placeholder commitments.",
        hint="An interval takes 12-24 hours, and this record's committed hashes "
        "are not the ones\nthe run produced -- so the replay cannot match, "
        "however correct it is.\n"
        "Pass --allow-simulated if you are deliberately exercising the "
        "plumbing rather than\nauditing the run.",
    )


def _committed_hash(
    ctx: recordmod.StepContext, ref: StepRef, args: argparse.Namespace, manifest
) -> str:
    """Which digest the replay must reproduce.

    The published state-hash log wins where the manifest names one: the
    commitment is a static fact about the run, and the API is explicit that it
    derives placeholders until the pipeline publishes. A disagreement is
    reported rather than resolved silently — it means either the API is still
    simulating, or two sources that must agree do not.
    """
    source = _state_hash_source(args, manifest)
    published = None
    if source:
        # The log is log-numbered; the record is not.
        published = commitmentsmod.load(source).get(ref.log)
        if published:
            echo(kv("committed hash", published, f"from {source.rsplit('/', 1)[-1]}"))

    def on_conflict(pub: str, api: str) -> None:
        warn("the record and the published log disagree about this step's committed hash.")
        echo(paint(f"    published log  {pub}", "dim"))
        echo(paint(f"    record         {api}", "dim"))
        echo(
            paint(
                "    Auditing against the published log: the commitment is a "
                "static fact about the run, and the record derives "
                "placeholders until its pipeline publishes.",
                "dim",
            )
        )

    chosen = commitmentsmod.reconcile(
        ref.log, published=published, from_api=ctx.committed_hash or None, on_conflict=on_conflict
    )
    _check_inclusion(ctx.proof, chosen)
    return chosen


def _check_inclusion(proof: commitmentsmod.Proof | None, committed: str) -> None:
    """Does the record's own proof reach the root the record published?

    Microseconds, and it runs before the download rather than after twelve
    hours. A proof that cannot reproduce its own root means two values from one
    source disagree, which is the kind of contradiction this tool exists to
    surface rather than replay past.
    """
    if proof is None:
        return
    if not commitmentsmod.verify_inclusion(proof):
        raise AuditError(
            "the record's inclusion proof does not reproduce its own segment root.",
            hint="Two values the record published disagree, so there is nothing "
            "here worth spending a replay on. Worth reporting rather than "
            "working around.",
        )
    # Not "verified": recomputing a path against a root supplied by the same
    # API shows self-consistency. See commitments.verify_inclusion.
    echo(
        kv(
            "inclusion",
            f"consistent with segment root {proof.root[:12]}",
            "" if proof.leaf == committed else "for the record's hash, not the one being audited",
        )
    )


def _fork_for(record, run: str, ctx: recordmod.StepContext):
    """The fork descriptor for this step's segment, if one applies.

    Forks live on the segment, not the step receipt, so this is a second call.
    It is worth making: without the descriptor the boundary interval replays
    under the previous segment's rules and mismatches on its first step, which
    reads as a divergence in the run rather than a missing flag.
    """
    if ctx.segment is None:
        return None
    try:
        fork = record.segment_fork(run, ctx.segment)
    except AuditError:
        # A record that cannot answer is not grounds to refuse: audit_replay
        # would still run, and a segment with no fork is the common case.
        return None
    if fork and fork.applies_to(ctx.predecessor.step):
        return fork
    return None


def _unit_from_step(
    ctx: recordmod.StepContext,
    ref: StepRef,
    args: argparse.Namespace,
    manifest=None,
    fork=None,
    committed: str | None = None,
) -> kitmod.Unit:
    """An interval unit assembled from the step API's mutable context.

    Interval units do not live in the trajectory: their predecessor can be
    another auditor's upload, which is exactly the thing a static file cannot
    describe.
    """
    genesis = ctx.predecessor.genesis if ctx.predecessor.is_initial_weights else None
    if genesis is not None and not ref.is_genesis:
        raise AuditError(
            f"the record describes {ref} as starting from the run's initial weights, "
            "but only the first audit does.",
            hint="Every other step starts from a checkpoint -- the run's own or "
            "another auditor's. Auditing this one from init would replay the "
            "wrong interval.",
        )
    if ref.is_genesis and genesis is None:
        raise AuditError(
            f"{ref} starts from the run's initialization, and the record does not describe it.",
            hint="`gs://<run>/ckpt/step_000000000/` was never written: the "
            "checkpoints start at 100 and the initial state is regenerated from "
            "the seed. A record that names it as a published checkpoint is "
            "describing bytes that do not exist. Update the record, or audit a "
            "later step.",
        )
    return kitmod.Unit(
        kind="interval",
        config_name=args.config_name or ctx.run,
        until_step=ref.log,
        state_hash=committed or ctx.committed_hash,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        # The step API names the predecessor because it is mutable (it may be
        # another auditor's upload). The shard root is not mutable, so the
        # manifest supplies it when the API does not repeat it.
        #
        # `--predecessor-uri` overrides both, for the case the record names a
        # location the auditor cannot read. It changes only WHERE the bytes are
        # fetched from, never WHAT they must hash to: `verify_predecessor` still
        # holds them to the record's digest and the unit's own expectations.
        checkpoint_uri=(
            # For a from-init step this is the run DESCRIPTOR the record names,
            # read for its meta.json alone. --predecessor-uri still overrides,
            # for a reader who holds their own copy of it.
            genesis.descriptor_uri
            if genesis is not None and not getattr(args, "predecessor_uri", None)
            else (
                getattr(args, "predecessor_uri", None)
                or ctx.predecessor.uri
                or (manifest.artifacts.checkpoint_uri(ctx.predecessor.step) if manifest else None)
            )
        ),
        gcs_root=ctx.gcs_root or (manifest.artifacts.shards if manifest else None),
        # Never both: audit_replay refuses --descriptor-checkpoint with
        # --from-init, and a fork cannot apply to the run's first step anyway.
        descriptor_uri=(
            None
            if genesis is not None
            else (fork.descriptor_checkpoint_uri if fork else ctx.descriptor_uri)
        ),
        expect_hash=committed or ctx.committed_hash,
        # The record may not know the predecessor yet (null until the relay
        # reaches this step); the numbering always does.
        predecessor_step=(
            ctx.predecessor.step if ctx.predecessor.step is not None else ref.predecessor_log
        ),
        from_init=genesis is not None,
        init_state_hash_uri=(genesis.init_state_hash_uri if genesis else None),
    )


def _resolve_unit(traj: kitmod.Trajectory, args: argparse.Namespace) -> kitmod.Unit:
    if args.config_name:
        return traj.find(args.config_name, args.kind)
    units = traj.init_units if args.kind == "init" else traj.units
    if len(units) == 1:
        return units[0]
    names = "\n".join(f"  --config-name {u.config_name}    {u.label}" for u in traj.units)
    raise AuditError(
        f"this trajectory has {len(traj.units)} units; name the one you mean.",
        hint=names,
    )


def _workdir(args: argparse.Namespace, unit: kitmod.Unit) -> Path:
    if getattr(args, "workdir", None):
        return Path(args.workdir).expanduser().resolve()
    slug = unit.config_name if unit.is_init else f"{unit.config_name}-{unit.until_step}"
    return Path.cwd() / f"audit-{slug}"


def _build_plan(args: argparse.Namespace, record=None, manifest=None):
    """Returns (plan, kit, step_context|None)."""
    if manifest is None:
        manifest = manifestmod.resolve(getattr(args, "manifest", None))
    k, traj = _resolve_kit(args, manifest)
    ctx = None
    ref = None
    if getattr(args, "step", None) is not None:
        if record is None:
            record = _resolve_record(args, manifest)
        run = args.run or (manifest.run if manifest else None) or traj.name
        # --step is the AUDIT number: it is what the web app shows and what the
        # claim is issued for. Everything on disk -- checkpoint directories,
        # state_hashes.jsonl, metrics.jsonl, --until-step -- is log-numbered,
        # one higher. Convert once, here.
        ref = StepRef.from_audit(args.step)
        ctx = record.step_context(run, ref.audit)
        committed = _committed_hash(ctx, ref, args, manifest)
        unit = _unit_from_step(
            ctx, ref, args, manifest, _fork_for(record, run, ctx), committed=committed
        )
    else:
        unit = _resolve_unit(traj, args)
    if unit.is_genesis and getattr(args, "checkpoint", None):
        raise AuditError(
            "--checkpoint is not supported for genesis audits; use --predecessor-uri "
            "to stage a descriptor with its initial-state commitment."
        )
    paths = plan_mod.KitPaths(k.kit_id)
    return (
        plan_mod.build(
            unit=unit,
            kit=k,
            workdir=_workdir(args, unit),
            venv=paths.venv,
            device=args.device,
            save_handoff=not getattr(args, "no_handoff", False),
            checkpoint=Path(args.checkpoint) if getattr(args, "checkpoint", None) else None,
            audit_step=(ref.audit if ref else None),
            run=(
                ctx.run
                if ctx
                else (getattr(args, "run", None) or (manifest.run if manifest else ""))
            ),
            extra_args=getattr(args, "extra", None) or [],
        ),
        k,
        ctx,
    )


# ── rendering ────────────────────────────────────────────────────────────────

_MARK = {
    doctor.PASS: (PASS, "green"),
    doctor.FAIL: (FAIL, "red"),
    doctor.WARN: (WARN, "yellow"),
    doctor.SKIP: (SKIP, "dim"),
}


def _render_checks(checks: list[doctor.Check], *, verbose: bool = False) -> bool:
    for c in checks:
        mark, color = _MARK[c.status]
        echo(f"{paint(mark, color)} {kv(c.name, c.value, c.note).lstrip()}")
        if c.fix and (c.blocking or verbose):
            for line in c.fix.splitlines():
                echo(paint(f"    {line}", "dim"))
    blocking = [c for c in checks if c.blocking]
    echo()
    if blocking:
        echo(paint(f"{len(blocking)} blocking issue(s).", "red"))
        return False
    warns = [c for c in checks if c.status == doctor.WARN]
    echo(paint("Ready." + (f" ({len(warns)} warning(s))" if warns else ""), "green"))
    return True


def _state_hash_source(args: argparse.Namespace, manifest) -> str | None:
    return getattr(args, "state_hashes", None) or (
        manifest.artifacts.state_hashes if manifest else None
    )


def _verify_predecessor(
    plan: plan_mod.Plan, args: argparse.Namespace, pred, manifest=None
) -> verifymod.Verdict:
    """Everything that must hold about the STARTING checkpoint, before the
    replay it would otherwise waste."""
    echo(rule("predecessor"))
    echo()
    if pred.is_crowd_provided and pred.step is None:
        pred = replace(pred, step=plan.unit.predecessor_step)
    # A fresh gate may replace the unpacked directory. Its old verdict must
    # not authorize partial output if conversion or verification fails.
    try:
        plan.workdir.predecessor_record.unlink(missing_ok=True)
    except OSError as exc:
        raise AuditError(f"could not invalidate the predecessor verdict: {exc}") from exc
    verdict = verifymod.gate(
        checkpoint=plan.checkpoint_path(),
        pred=pred,
        venv=plan.venv,
        state_hashes=_state_hash_source(args, manifest),
        device=plan.device,
        unpack_to=plan.workdir.verified_predecessor,
    )
    if verdict.checkpoint is not None:
        plan.checkpoint = verdict.checkpoint
    return verdict


def _describe(plan: plan_mod.Plan) -> None:
    u = plan.unit
    echo(kv("kit", plan.kit.kit_id))
    echo(kv("unit", u.label))
    echo(kv("committed hash", u.target_hash))
    echo(kv("device", plan.device))
    if u.mps_wall_seconds:
        echo(
            kv(
                "published cost",
                f"{u.mps_wall_seconds:.0f}s on MPS",
                f"{u.mps_peak_rss_gb} GB peak" if u.mps_peak_rss_gb else "",
            )
        )
    echo(kv("workdir", str(plan.workdir.root)))
    for note in plan.notes:
        echo()
        warn(note)


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_units(args: argparse.Namespace) -> int:
    k, traj = _resolve_kit(args, manifestmod.resolve(getattr(args, "manifest", None)))
    head("Trajectory", f"{traj.name} · kit {k.kit_id}")
    echo(kv("pretrain", traj.pretrain_commit[:12]))
    echo(kv("repop", traj.repop_commit[:12], "hashes and wheels pair"))
    echo()
    for u in traj.units:
        cost = (
            f"{u.mps_wall_seconds:.0f}s · {u.mps_peak_rss_gb} GB"
            if u.mps_wall_seconds
            else "cost unpublished"
        )
        what = u.kind if u.is_init else f"{u.kind} → step {u.until_step}"
        echo(
            kv(
                u.config_name,
                what,
                f"{cost} · verified on {', '.join(u.devices_verified)}",
                key_width=22,
            )
        )
    if traj.notes:
        echo()
        echo(paint(traj.notes, "dim"))
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    plan, _, ctx = _build_plan(args)
    head("Preflight", f"{plan.unit.label} · gensyn-audit {__version__}")
    checks = doctor.run_checks(plan, deep=not args.quick, ctx=ctx)
    return EXIT_OK if _render_checks(checks, verbose=args.verbose) else EXIT_ERROR


def cmd_plan(args: argparse.Namespace) -> int:
    plan, _, _ = _build_plan(args)
    head(f"Plan {ARROW} {plan.unit.label}", plan.kit.prefix)
    _describe(plan)
    echo()
    echo(rule("what `gensyn-audit run` executes"))
    echo()
    echo(plan.shell_script())
    echo()
    echo(
        paint(
            "Everything repop needs to match the trained run (REPOP_EXECUTION_MODE, the "
            "Hadamard flags, CUBLAS_WORKSPACE_CONFIG) is applied by audit_replay from the "
            "checkpoint's own meta.json — not set here. The wheels put pretrain and repop "
            "on the venv's path, and the pretrain wheel ships configs/, so there is no "
            "PYTHONPATH and no checkout.",
            "dim",
        )
    )
    return EXIT_OK


def cmd_install(args: argparse.Namespace) -> int:
    plan, k, _ = _build_plan(args)
    head("Install kit", k.kit_id)
    _provision(plan, k)
    echo()
    echo(paint(f"gensyn-audit run --kit {k.prefix} --config-name {plan.unit.config_name}", "dim"))
    return EXIT_OK


def _provision(plan: plan_mod.Plan, k: kitmod.Kit) -> None:
    """Stage, verify and install. Idempotent; safe to re-enter."""
    paths = plan_mod.KitPaths(k.kit_id)
    wheels = k.wheels_in_install_order(_PLATFORM_TAG)
    wanted = (*wheels, *(f for f in k.files if f.name.endswith(".json")))

    def on_file(entry: kitmod.KitFile, what: str) -> None:
        if what == "verified":
            echo(f"  {paint(PASS, 'green')} {entry.name}  {paint('sha256 ok', 'dim')}")
        elif what == "cached":
            echo(f"  {paint(PASS, 'green')} {entry.name}  {paint('cached', 'dim')}")
        else:
            echo(f"    {entry.name}  {paint(f'{entry.bytes / 1e6:.0f} MB', 'dim')}")

    kitmod.stage(k, paths.stage, only=wanted, on_progress=on_file)
    staged = {f.name: paths.stage / f.name for f in wanted}
    kitmod.provision(
        k,
        paths.venv,
        staged,
        platform_tag=_PLATFORM_TAG,
        on_step=lambda m: echo(paint(f"  {m}", "dim")),
    )
    info = kitmod.verify_build(paths.venv / "bin" / "python", k, device=plan.device)
    echo(
        f"  {paint(PASS, 'green')} repop {info['commit'][:12]} "
        f"{paint('matches kit.json · backends: ' + ', '.join(info['backends']), 'dim')}"
    )


def cmd_run(args: argparse.Namespace) -> int:
    if args.supervised:
        # Blocks until the parent has written `run.json` and closed the pipe;
        # see `runner.supervise` for why that order, and why the claim comes
        # this way rather than on the command line.
        handoff = runner.read_handoff()
        if not args.claim and handoff.get(runner.HANDOFF_CLAIM):
            args.claim = handoff[runner.HANDOFF_CLAIM]
    for line in brand.banner():
        echo(line)
    manifest = manifestmod.resolve(args.manifest)
    if manifest:
        echo(kv("manifest", manifest.source, manifest.run))
    record = _resolve_record(args, manifest) if (args.step is not None or args.claim) else None
    _warn_if_simulated(record, args)
    plan, k, ctx = _build_plan(args, record, manifest)
    head(f"Audit {ARROW} {plan.unit.label}", k.prefix)
    _describe(plan)
    if plan.is_genesis:
        echo()
        echo(
            "This is the run's first step. There is no checkpoint before it: the "
            "initial weights are regenerated from the published seed and compared "
            "against the run's own init hash before the replay starts."
        )
    if ctx and ctx.predecessor.is_crowd_provided:
        echo()
        echo(
            "Using another auditor's checkpoint. Before replay, its artifact integrity "
            "is checked and its training-state hash is reconstructed and matched "
            "against the run's published log."
        )

    if args.print_only:
        echo()
        echo(rule("command"))
        echo()
        echo(plan.shell_script())
        return EXIT_OK

    plan.workdir.create()
    verdict = None

    # Determine whether this invocation will replay before modifying the workdir.
    # Live replays and diagnostics from completed replays must keep their spill.
    prior = _completed_replay(plan)
    replaying = prior is None or args.restart

    echo()
    echo(rule("kit"))
    echo()
    _provision(plan, k)

    # Clear stale spill before the disk check, but only when starting a replay.
    if replaying and (freed := plan.workdir.clear_scratch()):
        echo(
            paint(
                f"  reclaimed {human_bytes(freed)} of offload spill from a previous attempt", "dim"
            )
        )

    if not args.skip_doctor:
        echo()
        echo(rule("preflight"))
        echo()
        if not _render_checks(doctor.run_checks(plan, deep=False, ctx=ctx, for_replay=replaying)):
            echo(paint("Re-run with --skip-doctor to start anyway.", "dim"))
            return EXIT_ERROR

    # Fetch and gate the predecessor only when starting a replay. A supervised
    # child reuses the verdict its parent saved in `predecessor.json`.
    prepared = _prepared_checkpoint(plan.workdir) if args.supervised else None
    if plan.needs_data and replaying and prepared is None:
        echo()
        echo(rule("checkpoint"))
        echo()
        if ctx is None:
            raise AuditError(
                "an interval unit needs a step context, which comes from the record.",
                hint="Use the complete command generated by the audit website.",
            )
        if plan.is_genesis:
            # Kilobytes, not gigabytes: the descriptor and the init commitment.
            # The start state itself is regenerated by the replay.
            descriptor_uri, init_hash_uri = plan.genesis_sources()
            fetch.fetch_genesis(
                descriptor_uri,
                init_hash_uri,
                plan.genesis_root(),
                expected_init_hash=(
                    ctx.predecessor.genesis.init_state_hash if ctx.predecessor.genesis else None
                ),
            )
            gated = False
        else:
            gated = _stage_predecessor(plan, args, ctx)
        # The unit's descriptor URI, not the receipt's: it carries the fork
        # looked up from the segment, which is where forks actually live.
        if plan.unit.descriptor_uri:
            fetch.fetch_descriptor(plan.unit.descriptor_uri, plan.descriptor_path())

        echo()
        if plan.is_genesis:
            echo(rule("predecessor"))
            echo()
            verdict = verifymod.gate_genesis(
                plan.genesis_descriptor_path(),
                init_hash=(plan.genesis_root() / "state_hash_init.txt").read_text().strip(),
            )
            _remember_predecessor(plan, verdict)
        elif not gated:
            verdict = _verify_predecessor(plan, args, ctx.predecessor, manifest)
            # The bundle goes only once the record that lets `--restart` do
            # without it is on disk. If that write failed, the bundle is the
            # only thing a restart could re-verify, so it stays.
            if _remember_predecessor(plan, verdict) and (
                freed := plan.workdir.consume_predecessor_bundle(verdict.bundle)
            ):
                echo(
                    paint(
                        f"  reclaimed {human_bytes(freed)}: the downloaded bundle is "
                        "unpacked and verified, and nothing reads it again",
                        "dim",
                    )
                )
    elif plan.needs_data and prepared is not None:
        plan.checkpoint = prepared
        echo()
        echo(kv("checkpoint", str(prepared), "gated by the run that started this one"))

    echo()
    echo(rule("replay"))
    echo()

    # A replay already done in this workdir must not be done again. Re-running
    # `gensyn-audit run` after a detached replay finishes is the documented way to
    # submit it, and spending another twelve hours instead would be the single
    # most expensive bug this tool could have. (Resolved above, before the
    # workdir was touched.)
    if not replaying:
        echo(
            paint(
                "  this workdir already holds a finished replay; reporting it "
                "rather than replaying",
                "dim",
            )
        )
        echo(paint(f"  started {prior.started_at}   pass --restart to run it again", "dim"))
        # This invocation's identity wins over the one saved by the run that
        # produced the replay: detaching without a claim and re-running with
        # one is the documented way to submit, so ignoring the new claim would
        # make the resume path useless for the case it exists for.
        for field, value in (
            ("claim", args.claim),
            ("machine", args.machine),
            ("handle", args.handle),
        ):
            if value:
                setattr(prior, field, value)
        runner.save_state(plan.workdir, prior)
        return _report(
            plan,
            k,
            prior,
            record,
            interrupted=False,
            verdict=verdict,
            next_command=_next_step_command(args),
        )

    (plan.workdir.handoff / convertmod.BUNDLE_NAME).unlink(missing_ok=True)
    if args.detach:
        state = runner.supervise(
            plan,
            _supervised_argv(args, plan.workdir.root),
            claim=args.claim,
            machine=args.machine,
            handle=args.handle,
        )
        echo(
            paint(f"{PASS} started detached", "green")
            + paint(f"   pid {state.supervisor_pid}", "dim")
        )
        echo(kv("screen", str(plan.workdir.cli_log)))
        echo(kv("replay log", str(plan.workdir.log)))
        echo()
        echo(
            paint(
                "  It replays, then reports, submits and uploads the hand-off on its "
                "own.\n  Nothing needs re-running. To watch it:",
                "dim",
            )
        )
        echo()
        echo(f"  gensyn-audit status --workdir {plan.workdir.root} --follow")
        return EXIT_OK

    timeout = runner.parse_duration(args.timeout) if args.timeout else None
    interrupted = timed_out = False

    import time as _time

    started = _time.monotonic()
    live = None if args.verbose else Live()
    label = plan.unit.label

    expected = plan.unit.mps_wall_seconds if plan.device == "mps" else None

    def tick() -> None:
        if live is not None:
            live.update(
                _status_block(
                    progress_mod.parse_tail(plan.workdir.log),
                    started,
                    label,
                    expected,
                    is_init=plan.unit.is_init,
                )
            )

    try:
        state, timed_out = runner.run_foreground(
            plan,
            claim=args.claim,
            machine=args.machine,
            handle=args.handle,
            on_line=(lambda line: echo(paint(f"  {line}", "dim"))) if args.verbose else None,
            on_tick=tick,
            timeout=timeout,
            supervisor_pid=os.getpid() if args.supervised else None,
        )
    except KeyboardInterrupt:
        interrupted = True
        state = runner.load_state(plan.workdir)
        runner.stop(state, plan.workdir, force=True)
    finally:
        if live is not None:
            # The verdict must not print underneath a stale bar.
            live.clear()

    return _report(
        plan,
        k,
        state,
        record,
        interrupted=interrupted,
        timed_out=timed_out,
        verdict=verdict,
        next_command=_next_step_command(args),
    )


def _stage_predecessor(plan: plan_mod.Plan, args: argparse.Namespace, ctx) -> bool:
    """Put the starting checkpoint where the gate will look, downloading only
    if this machine does not already hold those bytes.

    Returns True when the checkpoint it chose has *already passed the gate* --
    a `--restart` replaying from the directory this workdir unpacked and
    verified last time -- and False when the gate still has to run on it.
    """
    # The unit's URI, not the receipt's: it is the resolved one, so
    # --predecessor-uri and the manifest's fallback both reach the download.
    # Reading the receipt here meant the override changed where the
    # descriptor guard looked but not where the bytes came from -- doctor
    # passed and run then fetched from the record's location anyway.
    source_uri = plan.unit.checkpoint_uri or ctx.predecessor.uri
    if source_uri is None:
        # Nothing to fetch and nothing to derive: the record does not know
        # where this step starts because nobody has handed off to it yet.
        # It carries no `predecessor.step` either, so the manifest's
        # `<checkpoints>/step_<n>` fallback has no number to fill in.
        raise _not_published_yet(None)

    # A restart reuses the verified predecessor directory left after its bundle
    # was consumed. `--refetch` explicitly requests a fresh copy.
    if (
        getattr(args, "restart", False)
        and not args.refetch
        and not getattr(args, "predecessor_uri", None)
    ):
        verified = _restartable_predecessor(plan, ctx)
        if verified is not None:
            echo(
                f"  {paint(PASS, 'green')} this workdir already unpacked and verified "
                "this hand-off; replaying from it"
            )
            echo(kv("checkpoint", str(verified), "gated before the previous replay"))
            echo(
                paint(
                    "  Not downloaded again: the bundle was consumed once the gate "
                    "passed. Pass\n  --refetch to download and verify it afresh.",
                    "dim",
                )
            )
            plan.checkpoint = verified
            return True

    # The record names a crowd hand-off by its digest. If this machine packed
    # a bundle with that digest -- the usual case for an auditor taking the
    # next step after their own -- those are the bytes, wherever the record
    # keeps its copy. The gate below holds the file to the same digest, so
    # reusing it changes where the bytes come from and nothing about what
    # they must be. `--refetch` and `--predecessor-uri` both override.
    if (
        ctx.predecessor.is_crowd_provided
        and not args.refetch
        and not getattr(args, "predecessor_uri", None)
    ):
        known = handoffs_mod.find(ctx.predecessor.digest)
        if known is not None:
            packed_for = f"step {known.step}" if known.step is not None else "an earlier step"
            echo(
                f"  {paint(PASS, 'green')} this machine packed this hand-off "
                f"({packed_for}); using it in place"
            )
            echo(kv("hand-off", str(known.path), human_bytes(known.size)))
            echo(
                paint(
                    "  Not downloaded again: the record publishes the same digest, "
                    "and the digest\n  check below runs on this file exactly as it "
                    "would on a download.",
                    "dim",
                )
            )
            plan.checkpoint = known.path
            return False

    try:
        fetched = fetch.fetch_checkpoint(
            source_uri,
            plan.checkpoint_path(),
            plan.unit,
            expected_digest=ctx.predecessor.digest,
            force=args.refetch,
        )
        plan.checkpoint = fetched
    except AuditError as exc:
        # A named-but-absent predecessor is the relay's normal race, not a
        # broken setup: an auditor's hand-off is quarantined until the
        # verification service clears it, and only then does it appear in
        # the public bucket this reads from. Saying "does not exist" alone
        # sends people hunting for a permissions problem they don't have.
        if not ctx.predecessor.is_crowd_provided or "does not exist" not in str(exc):
            raise
        raise _not_published_yet(source_uri) from exc
    return False


def _restartable_predecessor(plan: plan_mod.Plan, ctx) -> Path | None:
    """The directory a `--restart` may replay from without re-fetching.

    It is `verified-predecessor/` when `predecessor.json` says the gate
    unpacked it from a bundle whose digest is the one the record publishes
    *now*, and the directory is still there. Anything less -- no record, a
    record from an anchor or a directory checkpoint, a digest the record has
    since changed, a directory that was cleaned up -- means None, and the
    predecessor is staged and gated from scratch.
    """
    if not ctx.predecessor.is_crowd_provided or not ctx.predecessor.has_bundle_digest:
        return None
    try:
        doc = json.loads(plan.workdir.predecessor_record.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict) or doc.get("bundle_digest") != ctx.predecessor.digest:
        return None
    verdict = doc.get("verdict")
    if not isinstance(verdict, dict) or verdict.get("tensor_state_commitment") != "verified":
        return None
    path = Path(str(doc.get("checkpoint") or ""))
    expected = plan.workdir.verified_predecessor
    try:
        if not path.is_dir() or path.resolve() != expected.resolve():
            return None
    except OSError:
        return None
    return path


def _remember_predecessor(plan: plan_mod.Plan, verdict) -> bool:
    """Keep the gate's verdict beside the workdir, for the reports that come
    later without re-running it: a supervised child, or a re-invocation on a
    workdir whose predecessor has since been cleaned up.

    For a packed hand-off it also keeps the digest the bundle was held to, so
    a `--restart` can replay from the unpacked directory after the bundle is
    consumed (`_restartable_predecessor`). Returns whether the record landed.
    """
    doc = {"verdict": verdict.record(), "checkpoint": str(plan.checkpoint_path())}
    if getattr(verdict, "bundle_digest", None):
        doc["bundle_digest"] = verdict.bundle_digest
    try:
        plan.workdir.create()
        plan.workdir.predecessor_record.write_text(json.dumps(doc, indent=2) + "\n")
    except OSError as exc:
        warn(f"could not save the predecessor verdict: {exc}")
        return False
    return True


def _prepared_checkpoint(workdir) -> Path | None:
    """The directory the parent of a supervised run gated and left for it.
    None when there is no such record, in which case the child stages the
    predecessor itself rather than replay from a directory nothing filled."""
    try:
        doc = json.loads(workdir.predecessor_record.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    path = doc.get("checkpoint") if isinstance(doc, dict) else None
    return Path(path) if path and Path(path).exists() else None


def _supervised_argv(args: argparse.Namespace, workdir: Path) -> list[str]:
    """This invocation, as the detached child should receive it.

    Three things change. The marker goes in, right after the subcommand rather
    than at the end: `--extra` is an argparse REMAINDER and would swallow
    anything appended after it. `--workdir` is replaced by the absolute path
    the parent resolved: the child must land in the directory that was just
    staged and gated for it, whatever it was spelled as and whatever the
    child's working directory turns out to be -- a relative or omitted
    `--workdir` re-resolved by the child once put the whole replay in a nested
    copy, out of sight of `status` and `stop`. And `--claim` comes out: the
    token goes to the child down a pipe (`runner.supervise`), not through an
    argv that any user on the machine can list and that the log echoes.
    """
    command, *rest = args.raw_argv
    kept: list[str] = []
    skip_value = False
    for i, tok in enumerate(rest):
        if skip_value:
            skip_value = False
            continue
        if tok == "--extra":
            # Everything from here on is audit_replay's, whatever it looks like.
            kept.extend(rest[i:])
            break
        if tok == "--detach":
            continue
        if tok in ("--workdir", "--claim"):
            skip_value = True
            continue
        if tok.startswith(("--workdir=", "--claim=")):
            continue
        kept.append(tok)
    return [command, "--supervised", "--workdir", str(workdir), *kept]


def _next_step_command(args: argparse.Namespace) -> str | None:
    """The command that continues from the hand-off this run produced.

    The record names the audit's own upload as the next step's predecessor,
    and (see `_stage_predecessor`) this machine will not download it again --
    so the whole cost of the next step is a claim in the web app and one
    command. Spell that command out rather than leave it to be inferred.

    The claim cannot be taken here: the record mints tokens through the web
    app and the CLI only receives one. Until it can, this is the closest thing
    to "keep going" the tool can offer.
    """
    raw = getattr(args, "raw_argv", None)
    if not raw or getattr(args, "step", None) is None:
        return None
    step, nxt = int(args.step), int(args.step) + 1
    command, *rest = raw
    kept: list[str] = []
    detached = False
    skip_value = False
    for tok in rest:
        if skip_value:
            skip_value = False
            continue
        if tok in ("--detach", "--supervised"):
            detached = True
            continue
        if tok in ("--restart", "--refetch"):
            continue
        if tok in ("--step", "--claim", "--workdir"):
            skip_value = True
            continue
        if tok.startswith(("--step=", "--claim=", "--workdir=")):
            continue
        kept.append(tok)
    # New flags go before what was kept: `--extra` may be in there, and it is
    # a REMAINDER that would take anything after it as audit_replay's.
    fresh = ["--step", str(nxt), "--claim", f"<claim for step {nxt}>"]
    if getattr(args, "workdir", None):
        wd = Path(args.workdir).expanduser()
        # Advance the step where the name spells it as its own number, and
        # only the last such place: `open-1b-102` -> `open-1b-103`, but the
        # `103` inside `open-1b-1030` is not the step and stays, as does the
        # `1` in `run-101`. A name that does not carry the step gets it added.
        hits = list(re.finditer(rf"(?<!\d){step}(?!\d)", wd.name))
        if hits:
            last = hits[-1]
            name = f"{wd.name[: last.start()]}{nxt}{wd.name[last.end() :]}"
        else:
            name = f"{wd.name}-{nxt}"
        fresh += ["--workdir", str(wd.with_name(name))]
    if detached:
        fresh.append("--detach")
    from .ui import render_command

    return render_command(["gensyn-audit", command, *fresh, *kept], indent="  ")


_PHASE_LABEL = {
    "starting": "starting up",
    "building": "building the model",
    "generating": "regenerating the initial state",
    "loading": "loading the checkpoint",
    "fetching": "fetching this interval's shards",
    "replaying": "replaying",
    "hashing": "hashing the final state",
    "done": "done",
    "failed": "done",
}


#: Wide enough for the longest label, so the counts column never shifts.
_PHASE_WIDTH = max(len(v) for v in _PHASE_LABEL.values()) + 2


#: Hold an elapsed-vs-expected bar here rather than letting it reach 100%: it
#: is a guess against a different machine's measurement, and a bar sitting full
#: while the work continues is a worse lie than one sitting at 95%.
_ESTIMATE_CAP = 0.95


def _reproduced_value(prog: progress_mod.Progress) -> str:
    """What to print for `reproduced`, never a bare `?`.

    A NO MATCH is actionable only with the reproduced digest. audit_replay
    emits only its 16-hex prefix on the mismatch path, so show that when the
    full digest is unavailable.
    """
    if prog.state_hash:
        return prog.state_hash
    if prog.state_hash_short:
        return f"{prog.state_hash_short}…"
    return paint("none produced", "yellow")


def _reproduced_note(prog: progress_mod.Progress) -> str:
    return (
        "as logged; audit_replay prints the full digest only on a match"
        if prog.reproduced_is_truncated
        else ""
    )


def _echo_verdict_hashes(
    prog: progress_mod.Progress, expect_hash: str, *, step: int | None = None
) -> None:
    """The committed/reproduced pair and the commitment's provenance."""
    echo(
        kv(
            "committed",
            expect_hash,
            f"published for step {step}" if step is not None else "published commitment",
        )
    )
    echo(kv("reproduced", _reproduced_value(prog), _reproduced_note(prog)))


def _echo_init_gate(prog: progress_mod.Progress, *, pending: bool = False) -> None:
    """The from-init gate, when the log carries one.

    A separate line because it is a separate check: the regenerated
    initialization against the published commitment, before any step replayed.
    Passing it says nothing about the interval -- the audit's verdict is still
    whatever the replay itself reports.
    """
    if prog.init_match is None:
        return
    echo(
        kv(
            "init gate",
            paint("passed", "green") if prog.init_match else paint("failed", "red"),
            "the audit verdict is separate and still pending"
            if pending
            else "regenerated initialization vs the published commitment",
        )
    )


def _status_block(
    prog: progress_mod.Progress,
    started: float,
    unit_label: str,
    expected_seconds: float | None = None,
    *,
    is_init: bool = True,
) -> list[str]:
    """Two lines: what it is doing, and how far through it is.

    Deliberately small. This is on screen for most of a day, and a wall of
    numbers that never changes is harder to read at a glance than a bar and a
    phase that do.
    """
    import time

    elapsed = time.monotonic() - started
    phase = _PHASE_LABEL.get(prog.phase, prog.phase)
    if prog.phase == "generating" and not is_init:
        # `param groups` is the last line before an init unit's long silent
        # generate-and-hash, which is what that label describes. An interval
        # logs the same line on its way into loading an 18 GB checkpoint, and
        # regenerating an initial state is not what it is doing.
        phase = "preparing the optimizer"

    fraction, estimate = prog.fraction, False
    right = human_duration(elapsed)

    if prog.phase in ("done", "failed"):
        # Complete is measured, not estimated. Without this the last frame
        # falls back to the indeterminate pulse and reads as going backwards.
        done_label = paint(f"{_PHASE_LABEL[prog.phase]:<{_PHASE_WIDTH}}", "cyan")
        return [
            f"  {done_label} {paint(unit_label, 'dim')}",
            f"  {bar(1.0)} 100%  {paint(human_duration(elapsed), 'dim')}",
        ]

    if prog.microbatches_total:
        counts = f"{prog.microbatches_done}/{prog.microbatches_total} microbatches"
        if prog.bar_eta and prog.bar_eta != "?":
            right += f" · ~{prog.bar_eta} left"
    else:
        # An init unit emits nothing between building the model and its digest.
        # Estimate progress from the duration published by the trajectory.
        counts = unit_label
        if expected_seconds and prog.phase not in ("done", "failed"):
            fraction = min(elapsed / expected_seconds, _ESTIMATE_CAP)
            estimate = True
            right += (
                f" · ~{human_duration(expected_seconds)} expected"
                if elapsed < expected_seconds
                else f" · over the ~{human_duration(expected_seconds)} expected"
            )

    if fraction is None:
        pct = "    "
    else:
        pct = f"{'~' if estimate else ' '}{int(fraction * 100):2d}%"
    # Pad the plain text and colour afterwards: an f-string width counts the
    # escape codes, so padding a painted string silently shortens the column by
    # however many bytes the colour cost.
    label = f"{phase:<{_PHASE_WIDTH}}"
    drawn = bar(fraction, tick=int(elapsed * 4), estimate=estimate)
    return [
        f"  {paint(label, 'cyan')} {paint(counts, 'dim')}",
        f"  {drawn} {pct}  {paint(right, 'dim')}",
    ]


def _completed_replay(plan: plan_mod.Plan) -> runner.RunState | None:
    """A finished replay in this workdir, if there is one.

    "Finished" means the process is gone AND the log carries a verdict. A
    workdir whose replay died without one is not finished — re-running should
    start over rather than report a crash as a result.
    """
    if not plan.workdir.state.is_file():
        return None
    state = runner.load_state(plan.workdir)
    if runner.supervisor_running(state):
        raise AuditError(
            f"a detached audit is already in progress here (pid {state.supervisor_pid}).",
            hint="It replays, reports, submits and uploads on its own; nothing needs "
            "re-running.\n"
            f"gensyn-audit status --workdir {plan.workdir.root} --follow\n"
            f"gensyn-audit stop   --workdir {plan.workdir.root}    # to abandon it",
        )
    if runner.is_running(state.pid):
        raise AuditError(
            f"a replay is already running here (pid {state.pid}).",
            hint=f"gensyn-audit status --workdir {plan.workdir.root}\n"
            f"gensyn-audit stop   --workdir {plan.workdir.root}    # to abandon it",
        )
    prog = progress_mod.parse_file(plan.workdir.log)
    if prog.match is None:
        return None
    # A detached replay records no finish of its own; without this the runtime
    # this invocation reports and submits would run to right now.
    return runner.settle_finish(state, plan.workdir)


def _runtime_text(state: runner.RunState) -> str:
    """How long the replay took, or "unknown" when no finish can be recovered."""
    if not state.finished_at:
        return paint("unknown", "yellow")
    try:
        elapsed = (
            datetime.fromisoformat(state.finished_at) - datetime.fromisoformat(state.started_at)
        ).total_seconds()
    except (TypeError, ValueError):
        return paint("unknown", "yellow")
    return human_duration(elapsed)


def _report(
    plan: plan_mod.Plan,
    k: kitmod.Kit,
    state: runner.RunState,
    record,
    *,
    interrupted: bool = False,
    timed_out: bool = False,
    verdict=None,
    next_command: str | None = None,
) -> int:
    prog = progress_mod.parse_file(plan.workdir.log)
    # The gate ran in this process, or in the one that prepared this workdir
    # (a supervisor's parent, or the run being re-reported). Either way the
    # receipt says what was established; never silence.
    established = _receipt_block(
        verdict.record() if verdict is not None else _prior_predecessor(plan.workdir)
    )
    outcome = classify(
        matched=prog.match,
        exit_code=state.exit_code,
        interrupted=interrupted,
        timed_out=timed_out or state.timed_out,
    )

    echo()
    echo(rule("result"))
    echo()

    if outcome is Outcome.MATCH:
        echo(paint(f"{PASS} MATCH \u2014 {outcome.headline}.", "green"))
    elif outcome is Outcome.NO_MATCH:
        echo(paint(f"{FAIL} NO MATCH \u2014 {outcome.headline}.", "red"))
    else:
        echo(paint(f"{WARN} {outcome.value.upper()} \u2014 {outcome.headline}.", "yellow"))

    # Provenance link 5, before any of this is reported as evidence.
    if prog.repop_commit:
        kitmod.verify_result_provenance({"repop": {"commit": prog.repop_commit}}, k)

    echo()
    _echo_verdict_hashes(prog, state.expect_hash, step=state.audit_step)
    _echo_init_gate(prog)
    if outcome is Outcome.NO_MATCH and prog.state_hash:
        diff = submit_mod.divergence(prog.state_hash, state.expect_hash)
        if diff:
            echo(kv("", paint(diff.display, "red")))
    echo(
        kv(
            "repop",
            prog.repop_commit[:12] if prog.repop_commit else "?",
            "matches kit.json" if prog.repop_commit else "",
        )
    )
    echo(kv("device", prog.device or state.device))
    if established and established.get("statement"):
        # Include the verified starting point alongside the final result.
        echo(kv("started from", established["statement"]))
    echo(kv("runtime", _runtime_text(state)))
    if prog.rank0_losses:
        last = prog.rank0_losses[-1]
        echo(
            kv(
                "losses",
                f"ce={last['loss_ce']!r} zloss={last['loss_zloss']!r}",
                f"{len(prog.rank0_losses)} step(s)",
            )
        )

    bundle = _pack_handoff(plan, state, prog, outcome)
    result = submit_mod.build(
        state,
        prog,
        plan.workdir,
        outcome=outcome,
        bundle=bundle,
        record_provenance=(record.provenance() if isinstance(record, mockmod.MockRecord) else None),
    )
    result.predecessor = established
    path = submit_mod.write(result, plan.workdir)
    echo()
    echo(kv("receipt", str(path)))

    code = _submit_and_upload(plan, state, result, record, outcome, next_command=next_command)
    if code is not None:
        return code

    if outcome is Outcome.NO_MATCH:
        echo()
        for line in NO_MATCH_EDUCATION.splitlines():
            echo(paint(f"  {line}", "dim"))
    elif outcome in (Outcome.INCONCLUSIVE, Outcome.TIMEOUT):
        echo()
        for line in INCONCLUSIVE_EDUCATION.splitlines():
            echo(paint(f"  {line}", "dim"))
        echo(paint(f"  {plan.workdir.log}", "dim"))
    return outcome.exit_code


def _not_published_yet(uri: str | None) -> AuditError:
    """The relay's normal race, told as one thing rather than two crashes.

    A hand-off is quarantined until the verification service clears it, and
    only then does it appear in the bucket this reads from. Until that happens
    the record publishes no URI and no predecessor step, so both the fetch and
    the manifest's `<checkpoints>/step_<n>` fallback have nothing to work with.
    """
    where = f": {uri}" if uri else "."
    return AuditError(
        f"this step's predecessor is not published yet{where}",
        hint="Position 1 of a segment starts from a published checkpoint and is\n"
        "always ready. Every later position starts from the previous\n"
        "auditor's upload, which stays quarantined until it passes the\n"
        "loss and digest checks. Re-check the step, or take one whose\n"
        "predecessor is already published:\n"
        "  a position-1 step is one where the audit step divides by 100.",
    )


def _pack_handoff(plan, state, prog, outcome) -> Path | None:
    """Pack before submission so the receipt declares the uploaded bytes.

    Conversion failure does not invalidate the completed replay result.
    """
    if state.unit_kind == "init" or not outcome.uploads_artifact:
        return None
    source = Path(prog.saved_checkpoint) if prog.saved_checkpoint else plan.workdir.handoff
    if not source.is_dir():
        return None  # audit_replay writes nothing when the hash did not match

    dest = plan.workdir.handoff / convertmod.BUNDLE_NAME
    if dest.is_file():
        return dest  # a resumed report re-uses the bundle it already packed
    echo()
    echo(rule("hand-off"))
    echo()
    try:
        with Activity(
            "packing the hand-off",
            detail="the converter re-reads every tensor to verify it; minutes at 18 GB",
        ):
            packed = convertmod.pack(plan.venv, source, dest)
    except AuditError as exc:
        warn(str(exc))
        for line in (exc.hint or "").splitlines():
            echo(paint(f"    {line}", "dim"))
        return None
    echo(kv("packed", human_bytes(packed.stat().st_size), packed.name))
    return packed


def _submit_and_upload(
    plan,
    state,
    result,
    record,
    outcome: Outcome,
    *,
    sidecar_only: bool = False,
    next_command: str | None = None,
) -> int | None:
    """Contact the record once, then upload if the match was accepted.

    Returns an exit code when it took over the reporting, else None.
    """
    if plan.unit.is_init:
        # The record's result endpoint is per-step and gates on that step's
        # losses. An init unit verifies the trajectory's starting point, not a
        # step, and advances no segment — submitting it would be asking the
        # wrong question of the wrong endpoint.
        echo()
        echo(
            paint(
                "An init unit is verified locally and is not submitted: it proves "
                "the run's starting state, not a step, so there is no segment for it "
                "to advance.",
                "dim",
            )
        )
        return None

    if record is None or not state.claim:
        echo()
        echo(
            paint(
                "Nothing was submitted: no claim token was given. The receipt above "
                "is the complete result bundle.",
                "dim",
            )
        )
        return None

    if not outcome.advances_record:
        # Still submitted: a failed replay is the input to the record's triage,
        # and it needs the machine, the runtime and the digest you actually got.
        echo()
        echo(paint(f"  submitting {outcome.value} for triage \u2014 nothing public changes", "dim"))

    import json as _json

    bundle = recordmod.result_bundle(outcome=outcome, receipt=_json.loads(result.to_json()))
    resumed = bool(state.submission_id) and state.submitted_with == state.claim
    if resumed:
        # The record already holds this result and the claim token is spent.
        # Resume the hand-off through its saved submission instead of posting again.
        response = recordmod.SubmitResponse(
            disposition="pending-verification",
            submission_id=state.submission_id,
            receipt_url=None,
            detail="",
            upload_required=True,
        )
        echo()
        echo(
            paint(f"{PASS} already recorded as {state.submission_id}", "green")
            + paint("   this token was spent by that submit; finishing the hand-off.", "dim")
        )
    else:
        response = record.submit(state.run, bundle, claim=state.claim)
        # Remember only a submission the record asked a hand-off for. A result
        # it recorded without asking (no digest declared, a no-match, an
        # already-accepted step) must not come back on the next run as a
        # pending upload and push a bundle the record never wanted. An
        # explicit `upload --submission` is the auditor's own call and stays
        # the override for that.
        if response.submission_id and response.recorded and response.upload_required:
            state.submission_id = response.submission_id
            state.submitted_with = state.claim
            try:
                runner.save_state(plan.workdir, state)
            except OSError as exc:
                warn(f"could not remember the submission id in run.json: {exc}")

    echo()
    if resumed:
        pass
    elif response.superseded:
        # Done and corroborating. Never a failure.
        echo(
            paint(f"{PASS} recorded — superseded", "green")
            + paint("   another auditor's match is already accepted; yours corroborates it.", "dim")
        )
    elif response.recorded:
        # Deliberately not "accepted": the loss gate runs asynchronously, and
        # telling an auditor their audit landed before it has been checked
        # would be the same overclaim the record's vocabulary rules forbid.
        echo(paint(f"{PASS} recorded — pending verification", "green"))
        echo(paint(f"   {response.detail}", "dim"))
        if response.verify_mode == "mock":
            echo(
                paint(
                    "   the verification service is mocked here, so an audit "
                    "accepted against it is flagged `simulated`.",
                    "yellow",
                )
            )
    else:
        echo(paint(f"{WARN} recorded, no public change", "yellow"))
        echo(paint(f"   {response.detail}", "dim"))

    if response.receipt_url:
        echo()
        echo(kv("receipt", response.receipt_url))
        echo(paint("   The outcome appears there once the loss gate reports.", "dim"))

    if (
        response.recorded
        and outcome.uploads_artifact
        and (response.upload_required or sidecar_only)
    ):
        uploaded = _do_upload(
            plan,
            state,
            record,
            response,
            result,
            sidecar_only=sidecar_only,
        )
        if (
            uploaded
            and outcome is Outcome.MATCH
            and not response.superseded
            and not record.is_mock
            and response.verify_mode != "mock"
        ):
            for line in brand.completion_banner():
                echo(line)
            if response.receipt_url:
                echo(kv("follow verification", response.receipt_url))
        audit_step = getattr(state, "audit_step", None)
        if uploaded and next_command and audit_step is not None:
            echo()
            echo(rule("next step"))
            echo()
            echo(
                paint(
                    f"  To be that auditor: claim step {audit_step + 1} in the web app, then", "dim"
                )
            )
            echo()
            echo(next_command)
            echo()
            echo(
                paint(
                    "  The hand-off you just uploaded is reused from this machine; "
                    "nothing is downloaded\n  again. Each step needs its own claim, so the "
                    "web app stays in the loop.",
                    "dim",
                )
            )
    elif response.recorded and outcome.uploads_artifact:
        # No artifactDigest was declared, so the record did not ask for an
        # upload. Say why once: the relay does not advance without a hand-off,
        # and an auditor who is not told will assume it did.
        _explain_missing_handoff()

    if outcome is Outcome.NO_MATCH:
        echo()
        for line in NO_MATCH_EDUCATION.splitlines():
            echo(paint(f"  {line}", "dim"))
    return outcome.exit_code


def _explain_missing_handoff() -> None:
    """Why a match produced no complete artifact for the next auditor."""
    echo()
    echo(rule("hand-off"))
    echo()
    warn("No artifact was offered to the record, so the relay does not advance past this step.")
    for line in (
        "The record accepts a safetensors hand-off of weights + optimizer with a",
        "handoff.json sidecar, and nothing else. audit_replay writes a",
        "torch.distributed.checkpoint directory plus gradients.safetensors and one",
        "batch-hasher chain per rank. These must be packed by the kit's",
        "pretrain-dcp-safetensors converter before submission. Check the",
        "packing error above or install a kit containing the current converter.",
        "",
        "Both matter to the recipient: without the gradients they cannot reconstruct",
        "the state hash the run published for this step, only trust that you did.",
        "",
        "Your result is recorded either way — this affects the next auditor, not you.",
    ):
        echo(paint(f"    {line}", "dim"))


def _record_run_id(record, result) -> str:
    """The canonical run id the verifier checks the sidecar against."""
    if record.is_mock:
        return result.run
    run_id = record.manifest().run_id
    if not run_id:
        raise AuditError(
            "the record manifest names no run id.",
            hint="handoff.json must carry the canonical id; sending the addressed "
            "run name would make the verifier reject this hand-off.",
        )
    return run_id


def _do_upload(
    plan,
    state,
    record,
    response,
    result,
    *,
    sidecar_only: bool = False,
) -> bool:
    """The bundle, then its sidecar. Return whether both reached intake.

    `sidecar_only` skips the bundle when the caller knows it already landed.
    Normally the saved upload session detects that itself; it is kept until the
    sidecar succeeds so a retry never sends the bundle again.
    """
    echo()
    echo(rule("hand-off"))
    echo()
    run_id = _record_run_id(record, result)
    try:
        source = Path(result.artifact.get("path") or plan.workdir.handoff)
        with Activity("digesting the hand-off", detail="BLAKE2b over the whole bundle"):
            bundle = uploadmod.build_bundle(source, plan.workdir.root)
    except AuditError as exc:
        # A missing hand-off must never turn a good audit into a failure. The
        # result is already on the record; only the relay artifact is absent,
        # and the next auditor is the one affected, not this one.
        warn(str(exc))
        for line in (exc.hint or "").splitlines():
            echo(paint(f"    {line}", "dim"))
        return False
    echo(kv("bundle", human_bytes(bundle.size), bundle.digest[:16]))
    # Before the upload, not after: a transfer that fails and is retried
    # tomorrow is still the same bytes, and the point is never to fetch them.
    handoffs_mod.remember(
        bundle.path, bundle.digest, step=getattr(state, "audit_step", None), run=state.run
    )
    ticket = record.upload_ticket(
        state.run,
        state.until_step,
        claim=state.claim,
        size=bundle.size,
        digest=bundle.digest,
        submission_id=(response.submission_id if response else None),
    )

    def on_progress(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 100
        print(f"\r  {pct:3d}%  {done / 1e9:.1f} / {total / 1e9:.1f} GB", end="", flush=True)

    if sidecar_only:
        echo(paint("  bundle skipped; sending handoff.json only", "dim"))
    else:
        uploadmod.send(bundle, ticket, on_progress=on_progress)
        print()
    uploadmod.send_sidecar(uploadmod.sidecar(result, bundle, run_id=run_id), ticket)
    uploadmod._forget_session(bundle)
    echo(kv("sidecar", "handoff.json", ticket.sidecar_uri or ""))
    if record.is_mock:
        problem = record.verify_upload(state.until_step)
        echo(
            f"  {paint(PASS, 'green') if not problem else paint(FAIL, 'red')} "
            f"{'digest verified by the record' if not problem else problem}"
        )
    echo(paint("  The next auditor starts from this artifact.", "dim"))
    return True


def _elapsed_origin(started_at: str) -> float:
    """A monotonic origin matching a wall-clock start recorded by another process.

    `_status_block` measures elapsed against `time.monotonic()`, which is only
    comparable within one process. A detached replay recorded its start as an
    ISO timestamp, so shift our own clock back by the wall-clock gap.
    """
    import datetime as _dt
    import time

    now = time.monotonic()
    try:
        began = _dt.datetime.fromisoformat(started_at)
    except (TypeError, ValueError):
        return now
    if began.tzinfo is None:
        began = began.replace(tzinfo=_dt.UTC)
    gap = (_dt.datetime.now(_dt.UTC) - began).total_seconds()
    return now - max(gap, 0.0)


def cmd_verify(args: argparse.Namespace) -> int:
    """`gensyn-audit verify` — the predecessor gate on its own.

    Same code path `run` takes, so what it reports is what `run` would enforce.
    Useful before committing a machine to a replay, and on a hand-off staged by
    hand rather than downloaded.
    """
    for line in brand.banner():
        echo(line)
    manifest = manifestmod.resolve(args.manifest)
    record = _resolve_record(args, manifest)
    plan, k, ctx = _build_plan(args, record, manifest)
    if ctx is None:
        raise AuditError(
            "verification needs the step whose predecessor this is.",
            hint="gensyn-audit verify --step <N> [--checkpoint <dir>]\n"
            "The step is what names the provenance and the published "
            "hashes; a directory on its own says only what its author "
            "wrote into it.",
        )
    head(f"Verify {ARROW} predecessor of {plan.unit.label}", k.prefix)
    echo()
    verdict = _verify_predecessor(plan, args, ctx.predecessor, manifest)
    _remember_predecessor(plan, verdict)
    echo()
    echo(
        paint(
            "  Nothing was replayed: this checks where the audit would START. "
            "The interval\n  itself is `gensyn-audit run`.",
            "dim",
        )
    )
    return EXIT_OK


def _guard_uploadable(root):
    """Is there a finished replay here to send? Returns (state, progress).

    Three refusals, each naming the command that actually helps: there is no
    audit here, the replay is still going, or it stopped without a verdict.
    None of them is fixed by retrying an upload.
    """
    wd = plan_mod.Workdir(Path(root))
    if not wd.state.is_file():
        raise AuditError(
            f"no audit in {wd.root}.",
            hint="`upload` sends the result of a replay that already finished, so "
            "run the audit first:\n"
            "  gensyn-audit run --manifest <url> --step <n> --claim <token>",
        )
    state = runner.load_state(wd)
    if runner.is_running(state.pid):
        raise AuditError(
            f"that replay is still running (pid {state.pid}).",
            hint=f"gensyn-audit status --workdir {wd.root} --follow",
        )
    # This path builds and sends a receipt, so the runtime it carries must be
    # the replay's, not the gap between the replay and this command.
    runner.settle_finish(state, wd)
    prog = progress_mod.parse_file(wd.log)
    if prog.match is None:
        raise AuditError(
            "that replay did not finish, so there is nothing to submit.",
            hint="Its log carries no verdict. Run the audit to completion first:\n"
            f"  gensyn-audit run --workdir {wd.root} …\n"
            "`upload` only re-sends a result that already exists.",
        )
    return state, prog


def _receipt_block(established: dict | None) -> dict | None:
    """The predecessor verdict as the receipt carries it: what was checked,
    not where on this machine the bytes were."""
    if not established:
        return None
    return {k: v for k, v in established.items() if k != "checkpoint"}


def _prior_predecessor(workdir) -> dict | None:
    """What the gate established about this workdir's starting checkpoint.

    From the verdict the gate saved when it last ran (`predecessor.json`, which
    also names the directory the replay starts from, under `checkpoint`), else
    from the receipt an earlier run wrote. The gate's own record wins because
    it is the fresher of the two after `--restart`. Either records a gate that
    ran once, before the replay; re-running it to report would demand a
    predecessor that may legitimately have been cleaned up since.
    """
    for path, key in (
        (workdir.predecessor_record, "verdict"),
        (workdir.root / "result.json", "predecessor"),
    ):
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        block = doc.get(key) if isinstance(doc, dict) else None
        if isinstance(block, dict) and block:
            if key == "verdict" and doc.get("checkpoint"):
                return {**block, "checkpoint": doc["checkpoint"]}
            return block
    return None


def cmd_upload(args: argparse.Namespace) -> int:
    """Submit and upload a replay that already finished.

    `run` does this itself, and re-running it on a finished workdir does it
    again -- that is the documented path. This exists for the shapes where that
    is the wrong tool:

      * the bundle reached intake and the sidecar PUT did not, which leaves a
        submission the verifier rejects as `missing_artifact` and which no
        amount of replaying fixes;
      * the replay finished without a claim, and re-running `run` rebuilds a
        plan and a kit to do nothing but send a file;
      * the upload failed on a connection that has since come back.

    It never replays. If the workdir holds no finished replay it says so and
    stops: the fix then is to run the audit, not to retry sending something
    that does not exist.
    """
    manifest = manifestmod.resolve(args.manifest)
    if manifest:
        echo(kv("manifest", manifest.source, manifest.run))
    record = _resolve_record(args, manifest)
    plan, _kit, _ctx = _build_plan(args, record, manifest)

    state, prog = _guard_uploadable(plan.workdir.root)

    # A later claim wins, as it does on a resumed `run`: finishing detached and
    # claiming afterwards is the case this command exists for.
    for field, value in (("claim", args.claim), ("machine", args.machine), ("handle", args.handle)):
        if value:
            setattr(state, field, value)
    submission = getattr(args, "submission", None)
    if submission:
        if not state.claim:
            raise AuditError(
                "--submission needs the claim token it was submitted with.",
                hint="Pass --claim as well.",
            )
        state.submission_id, state.submitted_with = submission, state.claim
    if args.claim or args.machine or args.handle or submission:
        runner.save_state(plan.workdir, state)

    outcome = classify(matched=prog.match, exit_code=state.exit_code, timed_out=state.timed_out)
    head(f"Upload {ARROW} {plan.unit.label}", str(plan.workdir.root))
    echo()
    echo(
        kv(
            "replay",
            outcome.value,
            f"finished {state.finished_at}" if state.finished_at else f"started {state.started_at}",
        )
    )
    echo(kv("reproduced", _reproduced_value(prog), _reproduced_note(prog)))

    bundle = _pack_handoff(plan, state, prog, outcome)
    result = submit_mod.build(
        state,
        prog,
        plan.workdir,
        outcome=outcome,
        bundle=bundle,
        record_provenance=(record.provenance() if isinstance(record, mockmod.MockRecord) else None),
    )
    # The predecessor gate ran during the replay, against bytes this command
    # does not require to still be on disk. Rebuilding the receipt without what
    # it established would not merely omit the claim -- it would overwrite the
    # run's own record of it with silence.
    result.predecessor = _receipt_block(_prior_predecessor(plan.workdir))
    if result.predecessor:
        echo(kv("started from", result.predecessor.get("statement", "")))
    submit_mod.write(result, plan.workdir)
    code = _submit_and_upload(plan, state, result, record, outcome, sidecar_only=args.sidecar_only)
    return code if code is not None else outcome.exit_code


def cmd_status(args: argparse.Namespace) -> int:
    workdir = plan_mod.Workdir(Path(args.workdir or Path.cwd()).expanduser().resolve())
    state = runner.load_state(workdir)
    prog = progress_mod.parse_file(workdir.log)
    alive = runner.is_running(state.pid)
    supervised = runner.supervisor_running(state)
    # `status` never writes: a workdir being kept as evidence stays untouched.
    runner.settle_finish(state, workdir, persist=False)

    head(state.label, f"kit {state.kit_id} · {workdir.root}")
    if prog.match is True:
        echo(paint(f"{PASS} MATCH", "green"))
    elif prog.match is False:
        echo(paint(f"{FAIL} NO MATCH", "red"))
    elif alive:
        echo(paint("● running", "cyan"))
    elif supervised:
        echo(
            paint("● preparing the replay", "cyan")
            + paint(f"   detached run, pid {state.supervisor_pid}", "dim")
        )
    else:
        echo(paint(f"{WARN} not running, no verdict", "yellow"))

    origin = _elapsed_origin(state.started_at)

    if args.follow and (alive or supervised) and prog.match is None:
        return _follow(workdir, state, origin)

    echo()
    if alive and prog.match is None:
        # The same block `run` draws when attached, so a detached replay is not
        # a worse view of the same work -- just a snapshot of it.
        echo()
        for line in _status_block(prog, origin, state.label, is_init=state.unit_kind == "init"):
            echo(line)
        echo()
        echo(paint("  --follow to watch it live", "dim"))
        echo()

    echo(kv("started", state.started_at))
    how = "detached" if state.detached else "attached"
    if state.pid:
        echo(kv("process", f"pid {state.pid}" + ("" if alive else " (exited)"), how))
    if state.detached and not alive and prog.match is not None:
        _echo_detached_aftermath(workdir, state, supervised)
    if prog.match is None:
        _echo_init_gate(prog, pending=True)
    if prog.fraction is not None and prog.match is None:
        echo(
            kv(
                "microbatches",
                f"{prog.microbatches_done}/{prog.microbatches_total}",
                f"eta {prog.bar_eta}" if prog.bar_eta else "",
            )
        )
    if live := progress_mod.read_loss_log(workdir.loss_log):
        echo(
            kv(
                "last step",
                str(live[-1]["step"]),
                f"ce={live[-1]['loss_ce']:.6f} zloss={live[-1]['loss_zloss']:.6g}",
            )
        )
    if prog.host_rss_gb is not None:
        echo(kv("memory", f"{prog.host_rss_gb:.1f} GB host"))
    if prog.match is not None:
        echo()
        _echo_verdict_hashes(prog, state.expect_hash, step=state.audit_step)
        _echo_init_gate(prog)
        if state.detached:
            if args.follow:
                _follow_supervisor(workdir, state)
            elif not (alive or supervised):
                _echo_supervisor_report(workdir, completion_only=True)
    return EXIT_MISMATCH if prog.match is False else EXIT_OK


def _echo_detached_aftermath(workdir, state, supervised: bool) -> None:
    """What became of a detached replay once it reached a verdict.

    Three honest states: the supervisor is still finishing (reporting,
    submitting, uploading); it finished and left a receipt; or it is gone
    without one, in which case the old recovery path is the answer.
    """
    if supervised:
        echo(paint(f"● reporting and submitting now   pid {state.supervisor_pid}", "cyan"))
        echo(paint(f"  its screen: {workdir.cli_log}", "dim"))
        return
    if workdir.result.is_file():
        echo(kv("receipt", str(workdir.result)))
        if state.submission_id:
            echo(kv("submitted", state.submission_id))
        if workdir.cli_log.is_file():
            echo(paint(f"  the full report, submission and upload: {workdir.cli_log}", "dim"))
        return
    if state.supervisor_pid:
        echo(
            paint(
                "  The detached run ended before it could report this. Re-run the same "
                "`gensyn-audit run`\n  to report and submit it; nothing is replayed.",
                "dim",
            )
        )
    else:
        echo(paint("  Re-run the same `gensyn-audit run` to report and submit it.", "dim"))


def _follow(workdir, state, origin: float) -> int:
    """Redraw the replay's progress until it stops or reaches a verdict.

    Polls the log rather than tailing the pipe: the replay is another process
    and its stdout belongs to whoever detached it. A supervised run is followed
    to the end of its submission as well, since that is when it is done.
    """
    import time

    echo()
    live = Live()
    try:
        while True:
            state = runner.load_state(workdir)  # the supervisor rewrites it
            prog = progress_mod.parse_file(workdir.log)
            alive = runner.is_running(state.pid)
            supervised = runner.supervisor_running(state)
            if not alive and prog.match is None and supervised:
                live.update(
                    [
                        f"  {paint('preparing the replay', 'cyan')}",
                        (
                            f"  {bar(None, tick=int(time.monotonic() * 4))}"
                            f"       {paint(f'pid {state.supervisor_pid}', 'dim')}"
                        ),
                    ]
                )
            else:
                live.update(
                    _status_block(prog, origin, state.label, is_init=state.unit_kind == "init"),
                    force=not alive or prog.match is not None,
                )
            if prog.match is not None or not (alive or supervised):
                break
            time.sleep(2.0)
    except KeyboardInterrupt:
        live.clear()
        echo()
        echo(paint("  Stopped watching. The replay is still running.", "dim"))
        return EXIT_OK

    live.clear()
    prog = progress_mod.parse_file(workdir.log)
    echo()
    if prog.match is True:
        echo(paint(f"{PASS} MATCH", "green"))
    elif prog.match is False:
        echo(paint(f"{FAIL} NO MATCH", "red"))
        echo()
        _echo_verdict_hashes(prog, state.expect_hash, step=state.audit_step)
    else:
        echo(paint(f"{WARN} the replay exited without a verdict", "yellow"))
        return EXIT_ERROR

    if runner.supervisor_running(state):
        _follow_supervisor(workdir, state)
    else:
        echo()
        _echo_detached_aftermath(workdir, state, False)
        _echo_supervisor_report(workdir, completion_only=True)
    return EXIT_MISMATCH if prog.match is False else EXIT_OK


def _follow_supervisor(workdir, state) -> None:
    """Wait for the detached run to finish reporting, then show what it did."""
    import time

    echo()
    live = Live()
    try:
        while runner.supervisor_running(state):
            live.update(
                [
                    (
                        f"  {paint('reporting and submitting', 'cyan')} "
                        f"{paint(str(workdir.cli_log), 'dim')}"
                    ),
                    f"  {bar(None, tick=int(time.monotonic() * 4))}",
                ]
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        live.clear()
        echo()
        echo(paint("  Stopped watching. The detached run is still finishing.", "dim"))
        return
    live.clear()
    _echo_supervisor_report(workdir)


def _echo_supervisor_report(workdir, *, completion_only: bool = False) -> None:
    """Render the saved report for this terminal without changing the log."""
    if not workdir.cli_log.is_file():
        return
    lines = _terminal_lines(workdir.cli_log.read_bytes())
    # Logs append across retries. Never promote an earlier attempt's completion
    # when the latest supervisor stopped before producing a result.
    start = max(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith(("result ", "===== supervisor started "))
        ),
        default=0,
    )
    report = "\n".join(lines[start:])
    plain = "\n".join(brand.completion_banner(plain=True))
    if completion_only:
        start = report.find(plain)
        if start < 0:
            return
        report = report[start:]
    echo(report.replace(plain, "\n".join(brand.completion_banner())))


def cmd_stop(args: argparse.Namespace) -> int:
    """Abandon a detached replay.

    Needed because `--detach` survives the terminal that started it: without
    this there is no way to end one short of finding the pid yourself.
    """
    workdir = plan_mod.Workdir(Path(args.workdir or Path.cwd()).expanduser().resolve())
    state = runner.load_state(workdir)
    if not runner.is_running(state.pid) and not runner.supervisor_running(state):
        echo(f"Nothing running here (pid {state.pid} has already exited).")
        return EXIT_OK
    what = (
        f"replay (pid {state.pid})"
        if runner.is_running(state.pid)
        else f"detached run (pid {state.supervisor_pid})"
    )
    if not args.yes and sys.stdin.isatty():
        answer = (
            input(f"Stop the {state.label} {what}? Its progress is lost. [y/N] ").strip().lower()
        )
        if answer not in ("y", "yes"):
            echo("Left running.")
            return EXIT_OK
    runner.stop(state, workdir, force=args.force)
    echo(
        paint(f"{PASS} stopped.", "green")
        + paint("  The claim lapses on its own; nothing needs releasing.", "dim")
    )
    return EXIT_OK


def _terminal_lines(raw: bytes) -> list[str]:
    """What a terminal would show, not what ``str.splitlines`` sees.

    A live progress bar redraws one line in place with a bare ``\r`` and only
    moves to the next with a real ``\n``. Two things turn that into one
    printed line per redraw -- tens of thousands of them for a 12-24h interval
    unit: ``str.splitlines`` treats a bare ``\r`` as its own break, and
    ``Path.read_text``'s universal-newline translation rewrites every ``\r``
    to ``\n`` before anything else gets a look, which is why the decode has to
    happen here, on bytes. Collapse each ``\r`` group to its last state, the
    way a terminal leaves it on screen.
    """
    # Universal-newline translation is the reason we are reading bytes, but
    # CRLF is still an ordinary line ending and not a redraw.
    text = raw.decode(errors="replace").replace("\r\n", "\n")
    segments = text.split("\n")
    # A trailing "\n" -- every real log has one -- leaves an empty segment
    # nobody wrote, as does an empty file. splitlines() never produced one.
    if segments[-1] == "":
        segments.pop()
    return [segment.rsplit("\r", 1)[-1] for segment in segments]


def cmd_logs(args: argparse.Namespace) -> int:
    workdir = plan_mod.Workdir(Path(args.workdir or Path.cwd()).expanduser().resolve())
    if not workdir.log.is_file():
        raise AuditError(f"no log at {workdir.log}.")
    lines = _terminal_lines(workdir.log.read_bytes())
    for line in lines[-args.lines :] if args.lines else lines:
        print(line)

    if args.follow:
        try:
            for chunk in runner.follow(workdir.log):
                sys.stdout.write(chunk)
                sys.stdout.flush()
        except KeyboardInterrupt:
            echo()
            echo(paint("Stopped watching. The replay is still running.", "dim"))
    return EXIT_OK


# ── parser ───────────────────────────────────────────────────────────────────


def _kit_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kit", help="kit prefix (gs://, https:// or a local dir)")
    p.add_argument("--config-name", help="which unit of the trajectory to verify")
    p.add_argument("--kind", default=None, choices=["init", "interval"], help="unit kind")
    p.add_argument("--device", default=_DEFAULT_DEVICE, choices=["mps", "cuda", "cpu"])
    p.add_argument("--workdir")
    p.add_argument(
        "--manifest",
        help="run manifest: names the kit, the record and the artifact "
        "roots, so no location is compiled into this tool",
    )
    p.add_argument("--record", help="record API base URL")
    p.add_argument("--run", help="run id the record is addressed by")
    p.add_argument(
        "--predecessor-uri",
        help="fetch the predecessor from here instead of where the "
        "record says; the digest gate is unchanged",
    )
    p.add_argument(
        "--state-hashes",
        help="published state_hashes.jsonl to take committed hashes "
        "from (overrides the record's, which may be derived)",
    )
    p.add_argument(
        "--step",
        type=int,
        help="audit this training step — an interval unit, resolved from "
        "the step API rather than the trajectory",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gensyn-audit",
        description="Verify one unit of a published auditable-training trajectory.",
        epilog="Exit codes: 0 match · 1 error · 2 the replay finished and did not match.",
    )
    parser.add_argument("--version", action="version", version=f"gensyn-audit {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    u = sub.add_parser("units", help="what this kit's trajectory can verify", parents=[common])
    u.add_argument("--kit")
    u.add_argument("--manifest")
    u.set_defaults(
        func=cmd_units,
        config_name=None,
        kind=None,
        device=_DEFAULT_DEVICE,
        workdir=None,
        record=None,
        run=None,
        step=None,
        manifest=None,
        state_hashes=None,
    )

    d = sub.add_parser("doctor", help="check this machine can run the unit", parents=[common])
    _kit_flags(d)
    d.add_argument("--quick", action="store_true", help="skip the venv import probe")
    d.add_argument("-v", "--verbose", action="store_true")
    d.set_defaults(func=cmd_doctor)

    pl = sub.add_parser("plan", help="print the command `run` would execute", parents=[common])
    _kit_flags(pl)
    pl.add_argument("--no-handoff", action="store_true")
    pl.add_argument("--checkpoint")
    pl.set_defaults(func=cmd_plan)

    i = sub.add_parser("install", help="stage and install the kit's wheels", parents=[common])
    _kit_flags(i)
    i.set_defaults(func=cmd_install)

    r = sub.add_parser("run", help="install, preflight, replay, compare, report", parents=[common])
    _kit_flags(r)
    r.add_argument("--claim", help="claim token from the record")
    r.add_argument("--machine", help='hardware, e.g. "M4 Pro · 24 GB" — published with the result')
    r.add_argument("--handle", help="the name inscribed on the record")
    r.add_argument("--checkpoint", help="a predecessor checkpoint you already hold")
    r.add_argument("--detach", action="store_true", help="run in its own session and return")
    r.add_argument(
        "--refetch",
        action="store_true",
        help="re-download the predecessor, even one this workdir already verified",
    )
    r.add_argument(
        "--allow-simulated",
        action="store_true",
        help="run against a record whose commitments are placeholders",
    )
    r.add_argument(
        "--restart",
        action="store_true",
        help="replay again even if this workdir already holds a result",
    )
    r.add_argument(
        "--timeout",
        help="give up after this long (e.g. 18h, 90m). Reports `timeout`, "
        "which changes nothing public.",
    )
    r.add_argument("--print-only", action="store_true", help="show the command, run nothing")
    # Set by `--detach` on the copy of this tool it starts, never by hand: it
    # says "the workdir was prepared by my parent; replay, then finish".
    r.add_argument("--supervised", action="store_true", help=argparse.SUPPRESS)
    r.add_argument("--skip-doctor", action="store_true")
    r.add_argument("--no-handoff", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true", help="stream the replay's output")
    r.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="everything after this is passed to audit_replay verbatim",
    )
    r.set_defaults(func=cmd_run)

    v = sub.add_parser(
        "verify",
        help="check a hand-off against the published state hash — no replay",
        parents=[common],
        description="Reconstruct a downloaded hand-off's committed training-state "
        "hash on disk and compare it with the run's published log. "
        "Seconds; `run` does the same before it replays.",
    )
    _kit_flags(v)
    v.add_argument(
        "--checkpoint",
        help="a hand-off directory you already hold (default: the "
        "one --step's predecessor resolves to)",
    )
    v.set_defaults(func=cmd_verify, device="cpu")

    s = sub.add_parser("status", help="what a replay is doing, and its verdict", parents=[common])
    s.add_argument("--workdir")
    s.add_argument("-f", "--follow", action="store_true", help="redraw until the replay finishes")
    s.set_defaults(func=cmd_status)

    up = sub.add_parser(
        "upload",
        help="submit and upload a replay that already finished",
        parents=[common],
        description="Send the result of a finished replay. `run` does this "
        "itself; reach for this when the replay is done and only "
        "the submission or the upload needs another go — a sidecar "
        "that failed after the bundle landed, a claim you did not "
        "have at the time, a connection that dropped. Never replays.",
    )
    _kit_flags(up)
    up.add_argument("--claim", help="claim token, if you did not have one when it ran")
    up.add_argument(
        "--submission",
        help="the submission id the record returned when this result was "
        "submitted, if this workdir does not remember it; the token "
        "was spent by that submit, so only the hand-off is sent",
    )
    up.add_argument(
        "--sidecar-only",
        action="store_true",
        help="the bundle already reached intake and only handoff.json "
        "failed; send the sidecar without re-sending 18 GB",
    )
    up.add_argument("--machine", help="what it ran on, for the public record")
    up.add_argument("--handle", help="how you want to be credited")
    up.set_defaults(func=cmd_upload)

    st = sub.add_parser("stop", help="abandon a detached replay", parents=[common])
    st.add_argument("--workdir")
    st.add_argument("--force", action="store_true", help="SIGKILL instead of SIGTERM")
    st.add_argument("-y", "--yes", action="store_true")
    st.set_defaults(func=cmd_stop)

    lg = sub.add_parser("logs", help="the replay's own output", parents=[common])
    lg.add_argument("--workdir")
    lg.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="keep watching — the counterpart to `run --detach`",
    )
    lg.add_argument("-n", "--lines", type=int, default=40, help="0 for the whole log")
    lg.set_defaults(func=cmd_logs)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw)
    # Kept verbatim: `--detach` hands this same invocation to a detached copy
    # of the tool, and the next-step hint is this invocation with the step
    # advanced. Rebuilding either from the parsed namespace would be a second
    # definition of the command line.
    args.raw_argv = raw
    if getattr(args, "no_color", False):
        set_plain(True)
    try:
        return args.func(args)
    except AuditError as exc:
        from .ui import fail

        echo()
        fail(str(exc), exc.hint)
        return EXIT_ERROR
    except KeyboardInterrupt:
        echo()
        echo(paint("Interrupted. Nothing was submitted.", "dim"))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
