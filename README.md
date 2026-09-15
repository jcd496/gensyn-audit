# gensyn-audit

`gensyn-audit` independently replays Gensyn Open 1B training steps and verifies
the results against the run's published commitments.

## Run an audit

The current Open 1B audit requires Python 3.11 and an Apple Silicon Mac.

Install the CLI:

```bash
uv tool install --python 3.11 gensyn-audit
gensyn-audit --version
```

Then go to:

https://open1b.gensyn.ai/

The site explains the available audits and their resource requirements, lets
you claim one, and gives you the exact command to run. Do not share its claim
token.

Run `gensyn-audit --help` for general CLI help.

## What it verifies

Depending on the selected audit, `gensyn-audit` checks the downloaded training
code and kernels, checkpoint integrity, and the replayed state against the
run's published commitments. Claimed results are also compared with loss values
withheld by the record.

A match means your machine independently reproduced the state hash Gensyn
published for that training step. Each successful audit verifies one step of
Open 1B's training history and contributes toward independently auditing the
full run.

## License

The audit client is open sourced under [Apache 2.0](./LICENSE).

Running an audit also uses RepOp, Gensyn's reproducibility
library, which is separately governed by the
[Gensyn Reproducibility License v1.0](./licenses/Gensyn-Reproducibility-License.md).
That license allows you to use RepOp to audit Open-1B's training and
publish your findings.
