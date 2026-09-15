"""Connect the app receipt to staging and the replay gate."""

import pytest
from test_genesis_step import INIT_HASH, STEP1_HASH, _args, _ctx, _plan, _publish

from gensyn_audit import cli, fetch, verify
from gensyn_audit.steps import StepRef


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
