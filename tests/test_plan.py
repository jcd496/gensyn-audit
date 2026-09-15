"""The runbook as data: what each unit kind actually runs, and with what env."""

from __future__ import annotations

from pathlib import Path

from conftest import INIT_HASH

from gensyn_audit import kit as K
from gensyn_audit import plan as plan_mod
from gensyn_audit import runner


def _plan(kit_dir: Path, tmp_path: Path, **kw) -> plan_mod.Plan:
    k = K.load_kit(str(kit_dir))
    traj = K.load_trajectory(k)
    unit = kw.pop("unit", None) or traj.find("1b_repop_run3", "init")
    return plan_mod.build(unit=unit, kit=k, workdir=tmp_path / "wd", venv=tmp_path / "venv", **kw)


def _interval_unit() -> K.Unit:
    return K.Unit(
        kind="interval",
        config_name="1b_repop_run3",
        until_step=801,
        state_hash="a" * 64,
        devices_verified=("mps",),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri="gs://b/step_000000800",
        gcs_root="gs://b/data/shards",
        expect_hash="c" * 64,
        predecessor_step=800,
    )


# ── init units ───────────────────────────────────────────────────────────────


def test_init_runs_the_console_script_not_a_module(kit_dir, tmp_path):
    """The kit's whole point: no checkout, so no `python -m` over a PYTHONPATH."""
    argv = _plan(kit_dir, tmp_path).argv()
    assert argv[0].endswith("/bin/pretrain-audit-replay")
    assert "-m" not in argv


def test_init_downloads_nothing_and_offloads_nothing(kit_dir, tmp_path):
    argv = _plan(kit_dir, tmp_path).argv()
    assert argv[1:] == [
        "--from-init",
        "--until-step",
        "0",
        "--config-name",
        "1b_repop_run3",
        "--device",
        "mps",
        "--expect-hash",
        INIT_HASH,
    ]
    for absent in ("--checkpoint", "--gcs-root", "--offload-optimizer", "--loss-log"):
        assert absent not in argv, absent


def test_init_needs_no_open_file_limit_and_no_data(kit_dir, tmp_path):
    p = _plan(kit_dir, tmp_path)
    assert p.is_init and not p.needs_data and not p.needs_open_files


def test_init_env_is_minimal(kit_dir, tmp_path):
    """An init unit allocates once and exits; the MPS memory knobs are noise."""
    env = _plan(kit_dir, tmp_path).env_overlay()
    assert env == {"WANDB_MODE": "disabled"}


def test_no_handoff_for_an_init_unit(kit_dir, tmp_path):
    """It regenerates from a seed and hands nothing to a next auditor."""
    p = _plan(kit_dir, tmp_path, save_handoff=True)
    assert not p.save_handoff
    assert "--save-checkpoint-dir" not in p.argv()


# ── interval units ───────────────────────────────────────────────────────────


def test_interval_asks_for_the_loss_log(kit_dir, tmp_path):
    """The record's gate runs on these, and the file survives a killed process
    where the result object does not."""
    p = _plan(kit_dir, tmp_path, unit=_interval_unit())
    argv = p.argv()
    assert argv[argv.index("--loss-log") + 1] == str(p.workdir.loss_log)


def test_interval_carries_the_runbook_flags(kit_dir, tmp_path):
    argv = _plan(kit_dir, tmp_path, unit=_interval_unit()).argv()
    for flag in (
        "--checkpoint",
        "--gcs-root",
        "--fetch-dest",
        "--until-step",
        "--expect-hash",
        "--offload-optimizer",
        "--offload-master",
        "--save-checkpoint-dir",
    ):
        assert flag in argv, flag
    assert argv[argv.index("--expect-hash") + 1] == "c" * 64, "the NEXT step's digest"
    assert argv[argv.index("--until-step") + 1] == "801"


def test_interval_env_carries_the_memory_knobs(kit_dir, tmp_path):
    env = _plan(kit_dir, tmp_path, unit=_interval_unit()).env_overlay()
    assert env["PRETRAIN_AUDIT_EMPTY_CACHE_PER_MB"] == "1"
    assert env["REPOP_INT8_MPP"] == "1"
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"
    # Applied from the checkpoint's own meta.json, never by hand.
    assert "REPOP_EXECUTION_MODE" not in env
    # The kit put both packages on the venv's path.
    assert "PYTHONPATH" not in env


def test_cpu_device_drops_the_mps_only_knobs(kit_dir, tmp_path):
    p = _plan(kit_dir, tmp_path, unit=_interval_unit(), device="cpu")
    assert "REPOP_INT8_MPP" not in p.env_overlay()
    assert "--offload-master" not in p.argv()


# ── environment hygiene ──────────────────────────────────────────────────────


def test_stray_pins_are_stripped_from_the_child(kit_dir, tmp_path, monkeypatch):
    """Both point at directories other than the ones inside the wheels, which
    is precisely what the kit exists to pin. An empty value is still an override."""
    monkeypatch.setenv("REPOP_METAL_SHADER_DIR", "/somewhere/stale")
    monkeypatch.setenv("PRETRAIN_CONFIGS", "")
    env = runner.build_env(_plan(kit_dir, tmp_path))
    assert "REPOP_METAL_SHADER_DIR" not in env
    assert "PRETRAIN_CONFIGS" not in env


def test_the_kit_venv_is_shared_across_audits(kit_dir, tmp_path):
    """~2GB of torch per kit, not per step."""
    a = plan_mod.KitPaths("pt-aaaaaaaaaaaa_rp-bbbbbbbbbbbb")
    b = plan_mod.KitPaths("pt-aaaaaaaaaaaa_rp-bbbbbbbbbbbb")
    assert a.venv == b.venv
    assert plan_mod.KitPaths("pt-cccccccccccc_rp-dddddddddddd").venv != a.venv
    assert (tmp_path / "wd") not in a.venv.parents


def test_a_device_the_trajectory_has_not_seen_is_flagged_not_refused(kit_dir, tmp_path):
    """The digests are device-independent by design, so it is still a match —
    just the first one on that backend."""
    p = _plan(kit_dir, tmp_path, device="cuda")
    assert any("not cuda" in n for n in p.notes)


def test_the_offload_spill_is_reclaimed_between_attempts(kit_dir, tmp_path):
    """A killed replay leaves ~18 GB of spill that no later run can use. Left
    in place it fails the disk preflight on the retry, so a stopped audit
    becomes an unstartable one."""
    p = _plan(kit_dir, tmp_path, unit=_interval_unit())
    p.workdir.create()
    (p.workdir.scratch / "optimizer").mkdir(parents=True)
    (p.workdir.scratch / "optimizer" / "moments.bin").write_bytes(b"x" * 5000)

    freed = p.workdir.clear_scratch()
    assert freed == 5000
    assert not p.workdir.scratch.exists()
    # Idempotent: nothing to reclaim is not an error.
    assert p.workdir.clear_scratch() == 0


def test_clearing_scratch_leaves_the_download_alone(kit_dir, tmp_path):
    """The checkpoint and shards are expensive and reusable; only the spill is
    disposable."""
    p = _plan(kit_dir, tmp_path, unit=_interval_unit())
    p.workdir.create()
    p.workdir.data.mkdir(parents=True, exist_ok=True)
    (p.workdir.data / "shard.bin").write_bytes(b"keep me")
    ckpt = p.workdir.checkpoint_dir(25700)
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "state_hash.txt").write_text("abc")
    p.workdir.scratch.mkdir(parents=True, exist_ok=True)
    (p.workdir.scratch / "spill").write_bytes(b"toss me")

    p.workdir.clear_scratch()
    assert (p.workdir.data / "shard.bin").exists()
    assert (ckpt / "state_hash.txt").exists()


# ── fork boundaries ──────────────────────────────────────────────────────────


def _fork_unit():
    u = _interval_unit()
    return K.Unit(**{**u.__dict__, "descriptor_uri": "gs://b/new/step_000050300"})


def test_a_boundary_interval_passes_the_descriptor(kit_dir, tmp_path):
    """Regression: the kit rewrite dropped --descriptor-checkpoint entirely, so
    a fork-crossing interval would have replayed under the previous segment's
    clipper and mismatched on its first step — a configuration error that reads
    as a divergence in the run."""
    p = _plan(kit_dir, tmp_path, unit=_fork_unit())
    argv = p.argv()
    assert "--descriptor-checkpoint" in argv
    assert argv[argv.index("--descriptor-checkpoint") + 1] == str(p.descriptor_path())
    assert any("fork boundary" in n for n in p.notes), "must say why"


def test_an_ordinary_interval_passes_no_descriptor(kit_dir, tmp_path):
    argv = _plan(kit_dir, tmp_path, unit=_interval_unit()).argv()
    assert "--descriptor-checkpoint" not in argv


def test_a_fork_applies_to_exactly_one_interval():
    """One replay carries one descriptor. Everything before the boundary is
    ordinary under the old rules; everything after records its own."""
    from gensyn_audit.record import Fork

    fork = Fork(boundary_step=50200, descriptor_checkpoint_uri="gs://b/new/step_000050300")
    assert fork.applies_to(50200)
    assert not fork.applies_to(50100)
    assert not fork.applies_to(50300)


def test_a_null_fork_parses_as_none():
    """`fork: null` is the common case — most segments straddle nothing."""
    from gensyn_audit.record import _parse_fork

    assert _parse_fork(None) is None
    assert _parse_fork({}) is None
    assert _parse_fork({"boundaryStep": 1}) is None, "no descriptor URI means no overlay"
    got = _parse_fork(
        {
            "boundaryStep": 50200,
            "descriptorCheckpointUri": "gs://b/new/step_000050300",
            "note": "clipper fix",
        }
    )
    assert got.boundary_step == 50200 and got.note == "clipper fix"


# ── a checkpoint already on disk is not fetched again ────────────────────────


def _unit_for(tmp_path):
    from gensyn_audit.kit import Unit

    return Unit(
        kind="interval",
        config_name="c",
        until_step=25701,
        state_hash="a" * 64,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri="gs://absent/ckpt/025700/",
        gcs_root=None,
        descriptor_uri=None,
        expect_hash="a" * 64,
        predecessor_step=25700,
    )


def test_a_partial_checkpoint_is_not_mistaken_for_a_complete_one(tmp_path, monkeypatch):
    """`state_hash.txt` is 65 bytes and lands early in a 19 GB transfer, so its
    presence says nothing about completeness. With no published digest to check
    it against, the fetch must run -- `download_prefix` skips what is already
    there at the right size, which is a real completeness check."""
    from gensyn_audit import fetch

    dest = tmp_path / "step_000025700"
    dest.mkdir()
    (dest / "state_hash.txt").write_text("b" * 64 + "\n")

    called = []
    monkeypatch.setattr(fetch, "_copy_tree", lambda *a, **k: called.append(a))
    monkeypatch.setattr(fetch, "verify_predecessor", lambda *a, **k: None)

    fetch.fetch_checkpoint(
        "gs://absent/ckpt/025700/", dest, _unit_for(tmp_path), expected_digest=None
    )
    assert called, "trusted a checkpoint whose completeness was never established"


def test_a_matching_published_digest_short_circuits(tmp_path, monkeypatch):
    from gensyn_audit import fetch

    dest = tmp_path / "step_000025700"
    dest.mkdir()
    (dest / "state_hash.txt").write_text("b" * 64 + "\n")

    called = []
    monkeypatch.setattr(fetch, "_copy_tree", lambda *a, **k: called.append(a))
    monkeypatch.setattr(fetch, "verify_predecessor", lambda *a, **k: None)

    fetch.fetch_checkpoint(
        "gs://absent/ckpt/025700/", dest, _unit_for(tmp_path), expected_digest="b" * 64
    )
    assert called == [], "re-fetched a checkpoint that matches the record's digest"


def test_a_wrong_digest_still_refetches(tmp_path, monkeypatch):
    from gensyn_audit import fetch

    dest = tmp_path / "step_000025700"
    dest.mkdir()
    (dest / "state_hash.txt").write_text("b" * 64 + "\n")

    called = []
    monkeypatch.setattr(fetch, "_copy_tree", lambda *a, **k: called.append(a))
    monkeypatch.setattr(fetch, "verify_predecessor", lambda *a, **k: None)

    fetch.fetch_checkpoint(
        "gs://absent/ckpt/025700/", dest, _unit_for(tmp_path), expected_digest="c" * 64
    )
    assert called, "local bytes disagreed with the published digest and were kept"


# ── resumed audits and sparse placeholders ───────────────────────────────────


def test_disk_check_discounts_what_is_already_fetched(tmp_path):
    """A stopped interval leaves ~27 GB of checkpoint and shards behind. Sizing
    the retry as though nothing were there is what made a stopped audit
    unstartable: the space they occupy is subtracted from free disk and then
    demanded again."""
    from gensyn_audit import doctor

    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint" / "shard.distcp").write_bytes(b"x" * 4096)
    assert doctor._already_on_disk_gb(tmp_path) > 0


def test_sparse_placeholders_are_not_counted_as_fetched_bytes(tmp_path):
    """fetch_interval places ~1700 sparse placeholder .bin files so the loader
    can open every reader. They report terabytes and occupy nothing; counting
    apparent size claimed 1701 GB was already on disk."""
    import os

    from gensyn_audit import doctor

    data = tmp_path / "data"
    data.mkdir()
    hole = data / "untouched.bin"
    with open(hole, "wb") as fh:  # 8 GB of nothing
        fh.truncate(8 * 1024**3)
    if os.stat(hole).st_blocks * 512 > 1024**3:
        import pytest as _p

        _p.skip("filesystem does not support sparse files")

    assert doctor._already_on_disk_gb(tmp_path) < 1.0


def test_the_headroom_checks_are_dropped_when_no_replay_will_start(tmp_path):
    """`run` on a workdir that already holds a finished replay reports it and
    keeps the offload spill. Sizing the machine for a replay nobody is about to
    start would then fail the report on the very bytes it exists to preserve.
    """
    from types import SimpleNamespace

    from gensyn_audit import doctor
    from gensyn_audit import kit as K
    from gensyn_audit.plan import Workdir

    unit = K.Unit(
        kind="interval",
        config_name="r",
        until_step=103,
        state_hash="a" * 64,
        devices_verified=(),
        mps_wall_seconds=None,
        mps_peak_rss_gb=None,
        checkpoint_uri=None,
        gcs_root=None,
        expect_hash="a" * 64,
        predecessor_step=102,
    )
    plan = SimpleNamespace(
        device="cpu", unit=unit, workdir=Workdir(tmp_path), kit=None, venv=tmp_path / "venv"
    )

    sizing = {"memory", "free disk"}
    replaying = {c.name for c in doctor.run_checks(plan, deep=False)}
    reporting = {c.name for c in doctor.run_checks(plan, deep=False, for_replay=False)}

    assert sizing <= replaying, "a real replay must still be sized"
    assert not (sizing & reporting)
    assert replaying - sizing == reporting, "nothing else may be skipped"
