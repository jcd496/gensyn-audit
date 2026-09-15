"""Connect the app receipt to staging and the replay gate."""

from types import SimpleNamespace

import pytest
from test_genesis_step import INIT_HASH, STEP1_HASH, _args, _ctx, _kit, _plan, _publish

from gensyn_audit import cli, fetch, verify
from gensyn_audit.errors import AuditError
from gensyn_audit.steps import StepRef


@pytest.mark.parametrize("command", [["plan"], ["run"], ["run", "--detach"]])
def test_genesis_rejects_checkpoint_override(tmp_path, monkeypatch, command):
    descriptor, init_uri = _publish(tmp_path / "bucket")
    ctx = _ctx(
        {
            "kind": "initial weights",
            "step": 0,
            "uri": None,
            "genesis": {
                "descriptorUri": descriptor,
                "initStateHashUri": init_uri,
                "initStateHash": INIT_HASH,
            },
        }
    )
    monkeypatch.setattr(cli, "_resolve_kit", lambda *a: (_kit(), SimpleNamespace(name="open-1b")))
    args = cli.build_parser().parse_args([*command, "--step", "0", "--checkpoint", descriptor])
    record = SimpleNamespace(step_context=lambda *a: ctx)
    with pytest.raises(AuditError, match="--checkpoint.*--predecessor-uri"):
        cli._build_plan(args, record=record)


@pytest.mark.parametrize("trailing_slash", [False, True])
def test_app_genesis_receipt_stages_the_descriptor_the_plan_uses(tmp_path, trailing_slash):
    descriptor, init_uri = _publish(tmp_path / "bucket")
    if trailing_slash:
        descriptor += "/"
    ctx = _ctx(
        {
            "kind": "initial weights",
            "step": 0,
            "uri": None,
            "genesis": {
                "descriptorUri": descriptor,
                "initStateHashUri": init_uri,
                "initStateHash": INIT_HASH,
            },
        }
    )
    unit = cli._unit_from_step(ctx, StepRef.from_audit(0), _args(), None, None, STEP1_HASH)
    plan = _plan(tmp_path, unit=unit)
    staged = fetch.fetch_genesis(
        *plan.genesis_sources(), plan.genesis_root(), expected_init_hash=INIT_HASH
    )
    assert unit.predecessor_step == 0
    assert staged == plan.genesis_descriptor_path()
    argv = plan.argv()
    assert argv[argv.index("--checkpoint") + 1] == str(staged)
    verify.gate_genesis(plan.genesis_descriptor_path(), init_hash=INIT_HASH)
