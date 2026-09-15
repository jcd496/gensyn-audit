"""The run's first step, which starts from no checkpoint at all.

`gs://gensyn-open-1b/ckpt/step_000000000/` holds no objects and never did: the
run's checkpoints start at 100, and its initial state is regenerated from the
published seed and committed as `ckpt/state_hash_init.txt`. On 2026-09-14 the
record offered audit step 0 anyway, and the auditor who took it installed the
kit, passed preflight and died on an empty prefix.

So audit 0 is a from-init interval: the same replay, the same hand-off, the
same submission, with the start state rebuilt rather than downloaded. These
tests pin the parts that make it that and not an init unit — which verifies a
hash and stops — and the parts that make it refuse rather than guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gensyn_audit import cli, doctor, fetch
from gensyn_audit import kit as K
from gensyn_audit import plan as plan_mod
from gensyn_audit import record as recordmod
from gensyn_audit import verify as verifymod
from gensyn_audit.errors import AuditError
from gensyn_audit.steps import StepRef

INIT_HASH = "16554a1119745f1b9bf4ac60b03226e5e9ca1247a2bd56169cd4c822fd457aed"
STEP1_HASH = "b151a0989960f85e9102cf5bd6e0f7b9eb88ae16cbdbd5aeded0d8a0ec6f5fe8"
DESCRIPTOR = "gs://gensyn-open-1b/ckpt/step_000000100"
INIT_URI = "gs://gensyn-open-1b/ckpt/state_hash_init.txt"


def _genesis_unit(**over) -> K.Unit:
    base = {
        "kind": "interval",
        "config_name": "1b_repop_v2",
        "until_step": 1,
        "state_hash": STEP1_HASH,
        "devices_verified": (),
        "mps_wall_seconds": None,
        "mps_peak_rss_gb": None,
        "checkpoint_uri": DESCRIPTOR,
        "gcs_root": "gs://gensyn-open-1b/data/shards",
        "expect_hash": STEP1_HASH,
        "predecessor_step": 0,
        "from_init": True,
        "init_state_hash_uri": INIT_URI,
    }
    return K.Unit(**{**base, **over})


def _kit() -> K.Kit:
    return K.Kit(
        kit_format=1,
        pretrain_commit="a" * 40,
        repop_commit="b" * 40,
        files=(),
        prefix="gs://kit",
        public_url_base="gs://kit",
    )


def _plan(tmp_path: Path, *, device: str = "cpu", unit: K.Unit | None = None) -> plan_mod.Plan:
    return plan_mod.build(
        unit=unit or _genesis_unit(),
        kit=_kit(),
        workdir=tmp_path / "wd",
        venv=tmp_path / "venv",
        device=device,
        audit_step=0,
    )


# ── numbering ────────────────────────────────────────────────────────────────


def test_only_audit_zero_is_genesis():
    assert StepRef.from_audit(0).is_genesis
    assert StepRef.from_audit(0).log == 1, "audit 0 replays log step 1"
    for n in (1, 99, 100, 25_700):
        assert not StepRef.from_audit(n).is_genesis


# ── the command ──────────────────────────────────────────────────────────────


def test_it_replays_from_init_to_the_first_logged_step(tmp_path):
    argv = _plan(tmp_path).argv()
    assert "--from-init" in argv
    assert argv[argv.index("--until-step") + 1] == "1"
    assert argv[argv.index("--expect-hash") + 1] == STEP1_HASH


def test_the_checkpoint_it_names_is_the_descriptor_not_a_predecessor(tmp_path):
    p = _plan(tmp_path)
    argv = p.argv()
    named = Path(argv[argv.index("--checkpoint") + 1])
    assert named == p.genesis_descriptor_path()
    assert named.parent == p.genesis_root(), (
        "audit_replay reads state_hash_init.txt from the PARENT of --checkpoint; "
        "flattening the layout skips the init comparison silently"
    )


def test_it_still_hands_off_and_logs_losses_like_any_interval(tmp_path):
    """Not an init unit: this one advances the relay and is submitted."""
    p = _plan(tmp_path)
    argv = p.argv()
    assert p.is_genesis and not p.is_init
    assert p.save_handoff and "--save-checkpoint-dir" in argv
    assert "--loss-log" in argv
    assert p.needs_data and "--gcs-root" in argv


def test_no_optimizer_offload_on_mps_because_from_init_refuses_it(tmp_path):
    """`audit_replay` raises NotImplementedError for --offload-optimizer with
    --from-init. Passing it anyway would fail hours in, after the kit and the
    data; the memory check below is what tells the auditor instead."""
    argv = _plan(tmp_path, device="mps").argv()
    assert not [a for a in argv if a.startswith("--") and "offload" in a]


def test_an_ordinary_interval_on_mps_still_offloads(tmp_path):
    ordinary = _genesis_unit(from_init=False, init_state_hash_uri=None, until_step=101)
    argv = _plan(tmp_path, device="mps", unit=ordinary).argv()
    assert "--offload-optimizer" in argv and "--offload-master" in argv


def test_a_fork_descriptor_is_never_combined_with_from_init(tmp_path):
    """audit_replay rejects the combination outright, and no fork can apply to
    the first step of a run in any case."""
    argv = _plan(tmp_path, unit=_genesis_unit(descriptor_uri="gs://x/step_000050300")).argv()
    assert "--descriptor-checkpoint" not in argv


# ── staging ──────────────────────────────────────────────────────────────────


def _publish(root: Path, *, init_hash: str = INIT_HASH) -> tuple[str, str]:
    """A local stand-in for the published bucket layout."""
    ckpt = root / "ckpt" / "step_000000100"
    ckpt.mkdir(parents=True)
    (ckpt / "meta.json").write_text(json.dumps({"step": 100, "seed": 42}))
    (ckpt / "global_stream.json").write_text("{}")
    (ckpt / "state_hash.txt").write_text("c" * 64 + "\n")
    (root / "ckpt" / "state_hash_init.txt").write_text(init_hash + "\n")
    return str(ckpt), str(root / "ckpt" / "state_hash_init.txt")


def test_staging_fetches_kilobytes_and_puts_them_where_the_replay_looks(tmp_path):
    descriptor, init_uri = _publish(tmp_path / "bucket")
    dest = tmp_path / "wd" / "genesis"

    staged = fetch.fetch_genesis(descriptor, init_uri, dest, expected_init_hash=INIT_HASH)

    assert (staged / "meta.json").is_file()
    assert (staged / "global_stream.json").is_file(), (
        "the interval fetcher requires the file to exist even though from-init "
        "resets the stream to the origin"
    )
    assert (dest / "state_hash_init.txt").read_text().strip() == INIT_HASH
    # Nothing else: no tensors, no dcp/, no 19 GB.
    assert not (staged / "dcp").exists()


def test_it_refuses_when_the_record_and_the_published_init_hash_disagree(tmp_path):
    descriptor, init_uri = _publish(tmp_path / "bucket")
    with pytest.raises(AuditError, match="disagree about the run's init hash"):
        fetch.fetch_genesis(
            descriptor, init_uri, tmp_path / "wd" / "genesis", expected_init_hash="f" * 64
        )


def test_it_refuses_an_init_commitment_that_is_not_a_digest(tmp_path):
    descriptor, init_uri = _publish(tmp_path / "bucket", init_hash="not a hash")
    with pytest.raises(AuditError, match="hex digest"):
        fetch.fetch_genesis(descriptor, init_uri, tmp_path / "wd" / "genesis")


# ── the gate ─────────────────────────────────────────────────────────────────


def test_the_verdict_says_the_replay_does_the_checking(tmp_path):
    descriptor, init_uri = _publish(tmp_path / "bucket")
    dest = tmp_path / "wd" / "genesis"
    staged = fetch.fetch_genesis(descriptor, init_uri, dest)

    verdict = verifymod.gate_genesis(staged, init_hash=INIT_HASH)
    block = verdict.record()

    assert block["predecessor_provenance"] == recordmod.INIT
    assert block["artifact_integrity"] == "not-applicable", "nothing was downloaded to check"
    assert block["tensor_state_commitment"] == "verified-by-replay"
    assert "regenerated" in block["statement"] and "init hash" in block["statement"]


def test_the_gate_refuses_a_start_with_no_published_init_commitment(tmp_path):
    descriptor, init_uri = _publish(tmp_path / "bucket")
    staged = fetch.fetch_genesis(descriptor, init_uri, tmp_path / "wd" / "genesis")
    (staged.parent / "state_hash_init.txt").unlink()

    with pytest.raises(AuditError, match="no published init commitment"):
        verifymod.gate_genesis(staged, init_hash=INIT_HASH)


def test_the_artifact_gate_refuses_to_handle_a_from_init_predecessor(tmp_path):
    """Two gates, kept apart: `gate` holds downloaded bytes to a published
    digest, and there are no bytes here."""
    pred = recordmod.Predecessor(source="initial weights", uri=None, digest=None, step=0)
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(AuditError, match="no artifact for this gate"):
        verifymod.gate(checkpoint=tmp_path / "ckpt", pred=pred, venv=tmp_path, state_hashes=None)


# ── what the record must say ─────────────────────────────────────────────────


def _ctx(pred: dict) -> recordmod.StepContext:
    return recordmod._parse_step_context(
        "open-1b", 0, {"step": 0, "committed": STEP1_HASH, "predecessor": pred}
    )


def _args(**over):
    base = {"config_name": None, "predecessor_uri": None, "run": "open-1b"}
    return SimpleNamespace(**{**base, **over})


def test_a_genesis_record_produces_a_from_init_unit():
    ctx = _ctx(
        {
            "kind": "initial weights",
            "step": 0,
            "uri": None,
            "genesis": {
                "descriptorUri": DESCRIPTOR,
                "initStateHashUri": INIT_URI,
                "initStateHash": INIT_HASH,
            },
        }
    )
    unit = cli._unit_from_step(ctx, StepRef.from_audit(0), _args(), None, None, STEP1_HASH)

    assert unit.is_genesis
    assert unit.until_step == 1, "audit 0 targets log step 1"
    assert unit.checkpoint_uri == DESCRIPTOR
    assert unit.init_state_hash_uri == INIT_URI


def test_step_zero_is_refused_when_the_record_still_calls_it_a_checkpoint():
    """The state before this PR: the record named ckpt/step_000000000/, which
    holds no objects. Refusing beats downloading nothing for twenty minutes."""
    ctx = _ctx(
        {
            "kind": "published checkpoint",
            "step": 0,
            "uri": "gs://gensyn-open-1b/ckpt/step_000000000/",
        }
    )
    with pytest.raises(AuditError, match="initialization"):
        cli._unit_from_step(ctx, StepRef.from_audit(0), _args(), None, None, STEP1_HASH)


def test_a_later_step_is_refused_if_the_record_calls_it_from_init():
    ctx = _ctx(
        {
            "kind": "initial weights",
            "step": 100,
            "uri": None,
            "genesis": {"descriptorUri": DESCRIPTOR, "initStateHashUri": INIT_URI},
        }
    )
    with pytest.raises(AuditError, match="only the first audit"):
        cli._unit_from_step(ctx, StepRef.from_audit(100), _args(), None, None, STEP1_HASH)


def test_a_half_described_genesis_is_refused_at_parse_time():
    with pytest.raises(AuditError, match="init commitment"):
        _ctx({"kind": "initial weights", "step": 0, "genesis": {"descriptorUri": DESCRIPTOR}})


# ── preflight ────────────────────────────────────────────────────────────────


def test_preflight_names_the_higher_memory_floor_for_a_from_init_replay(monkeypatch):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: 36.0)
    check = doctor._check_memory(_genesis_unit(), "mps")

    assert check.status == doctor.FAIL, (
        "with no optimizer offload the moments stay resident; a machine that "
        "runs a later step happily cannot run this one"
    )
    assert "offload" in check.note


def test_preflight_passes_the_genesis_step_on_a_large_machine(monkeypatch):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: 64.0)
    assert doctor._check_memory(_genesis_unit(), "mps").status == doctor.WARN


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize("host", [None, 24.0, 36.0])
def test_genesis_rejects_insufficient_or_unknown_host_memory(monkeypatch, device, host):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: host)
    assert doctor._check_memory(_genesis_unit(), device).blocking


@pytest.mark.parametrize(
    "free, status",
    [(None, doctor.FAIL), (24.0, doctor.FAIL), (47.9, doctor.FAIL), (64.0, doctor.WARN)],
)
def test_genesis_cuda_checks_free_vram_separately(monkeypatch, tmp_path, free, status):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: 128.0)
    monkeypatch.setattr(K, "venv_python", lambda venv: venv / "python")
    (tmp_path / "python").touch()
    seen = []

    def probe(venv):
        seen.append(venv)
        return free

    monkeypatch.setattr(doctor, "_cuda_free_gb", probe)
    assert doctor._check_memory(_genesis_unit(), "cuda", tmp_path).status == status
    assert seen == [tmp_path]


@pytest.mark.parametrize("venv", [None, "missing"])
def test_genesis_cuda_defers_the_vram_check_until_the_kit_exists(monkeypatch, tmp_path, venv):
    """`doctor` runs before the kit is installed; that is not a failed card.

    The probe needs the kit's torch. Failing here would tell everyone on CUDA
    that their machine cannot audit step 0 for the sole reason that they ran
    the check first, which is what the check is for. `run` installs the kit
    and repeats the check, so the deferral costs nothing.
    """
    monkeypatch.setattr(doctor, "_memory_gb", lambda: 128.0)
    monkeypatch.setattr(doctor, "_cuda_free_gb", lambda venv: pytest.fail("probed a missing kit"))
    check = doctor._check_memory(_genesis_unit(), "cuda", None if venv is None else tmp_path / venv)
    assert check.status == doctor.WARN
    assert not check.blocking
    assert "not yet measured" in check.value
    assert "run" in check.fix


def test_genesis_cpu_warns_even_when_above_the_provisional_floor(monkeypatch):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: 64.0)
    check = doctor._check_memory(_genesis_unit(), "cpu")
    assert check.status == doctor.WARN
    assert "not a measured" in check.note


@pytest.mark.parametrize("output", ["nan", "inf", "-1", "not available", ""])
def test_cuda_probe_refuses_invalid_measurements(monkeypatch, tmp_path, output):
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setattr(
        doctor, "_run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=output)
    )
    assert doctor._cuda_free_gb(tmp_path) is None


def test_cuda_probe_uses_kit_python_and_preserves_device_selection(monkeypatch, tmp_path):
    python = tmp_path / "bin" / "python"
    python.parent.mkdir()
    python.touch()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    def probe(argv, timeout):
        import os

        assert argv[0] == str(python)
        assert "mem_get_info" in argv[2]
        assert timeout == 30
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
        return SimpleNamespace(returncode=0, stdout="64.0\n")

    monkeypatch.setattr(doctor, "_run", probe)
    assert doctor._cuda_free_gb(tmp_path) == 64.0


def test_preflight_accepts_the_record_describing_a_from_init_predecessor():
    ctx = _ctx(
        {
            "kind": "initial weights",
            "step": 0,
            "genesis": {"descriptorUri": DESCRIPTOR, "initStateHashUri": INIT_URI},
        }
    )
    checks = doctor._check_predecessor_digest(ctx)
    assert [c.status for c in checks] == [doctor.PASS]
    assert "regenerated" in checks[0].note


def test_preflight_asks_for_less_disk_than_an_interval(tmp_path):
    """No predecessor download: the ~19-26 GB an ordinary step spends on one
    is not spent here."""
    genesis = doctor._check_disk(tmp_path, _genesis_unit())
    interval = doctor._check_disk(tmp_path, _genesis_unit(from_init=False))
    assert genesis.note != interval.note
