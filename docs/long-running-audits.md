# Long-running audits

An Open 1B interval step takes hours. This guide covers what the CLI does
while it runs, how to watch or abandon it, and what a low-memory machine is in
for. The README stays short because it doubles as the PyPI package
description; the audit website is the place to start.

## The detached flow

The command the site gives you runs detached. The CLI prepares everything in
front of you: kit, preflight, download, predecessor check. Anything that fails
fast fails while you are watching. It then hands the rest to a copy of itself
in its own session. That copy replays, reports, submits under your claim and
uploads the hand-off on its own. Closing the terminal does not stop it, and
nothing needs re-running.

```bash
gensyn-audit status --workdir <dir> --follow   # watch it through to the submission
gensyn-audit logs   --workdir <dir>            # the replay's own output
gensyn-audit stop   --workdir <dir>            # abandon it
```

Its screen is `<dir>/gensyn-audit.log`; the replay's own output is
`<dir>/audit.log`. The claim token is handed to the detached copy over a pipe,
not on its command line, so it does not appear in either log or in the
process list.

If the detached copy dies before reporting (a reboot, say), `status` says so.
Re-running the same `run` command reports and submits the finished replay
rather than starting another.

## The next step

When the upload finishes, the CLI prints the command for the next step. Claim
that step on the site and run the printed command. The hand-off you just
uploaded is reused from this machine rather than downloaded again; it is still
digest-checked against what the record published before anything opens it.
Every step needs its own claim.

## The run's first step

Audit step 0 is the only one that starts from no checkpoint. The run's
checkpoints begin at step 100, and the state before the first update is the
initialization: regenerated from the published seed, hashed, and published as
`ckpt/state_hash_init.txt`. `ckpt/step_000000000/` was never written.

So that step replays with `--from-init`. The tool downloads a few kilobytes —
the run descriptor's `meta.json` and `global_stream.json`, and the published
init hash — instead of a 19 GB predecessor, rebuilds the initial weights from
the seed, and refuses to replay unless they reproduce the published init hash.
Everything after that is an ordinary audit: the same replay, the same hand-off,
the same submission.

Two differences are worth knowing before you claim it:

- **Its memory requirements are provisional.** From-init cannot offload the
  optimizer. Preflight requires at least 48 GiB host memory for CPU and MPS;
  CUDA additionally requires 48 GiB free VRAM on the selected GPU. Unknown
  capacity is refused. These are screening thresholds, not measured sufficient
  capacities; passing them still produces a warning. The CUDA probe runs in
  the kit environment and respects `CUDA_VISIBLE_DEVICES`, so `doctor` on a
  machine without the kit reports VRAM as not yet measured rather than
  failing; `run` installs the kit, then checks, then fetches.
- **It downloads less.** No predecessor checkpoint, so the disk budget is
  smaller than an ordinary interval's: the shards the first step consumes, the
  hand-off it writes, and the kit.

Step-0 runtime and peak memory are not measured yet. Ordinary interval timings
below do not establish either for this path. Before public enablement, validate
the real first update, upload and fresh-download its handoff, reconstruct the
state hash and replay the next update. Exercise the real loss verifier in an
isolated/staging setup; simulated staging acceptance is not that proof.

## Memory and swap

Observed runtimes for a step are roughly:

| Machine                | Time per step   |
|------------------------|-----------------|
| 24 GB MacBook Pro      | about 18 hours  |
| 48 GB MacBook Pro      | about 6 hours   |
| H100                   | under an hour   |

There is no minimum spec. The point is that an audit is doable on ordinary
hardware. Below about 40 GB of unified memory the replay lives in swap, so
expect the long form. `doctor` warns about this rather than refusing.

If a heavily swapping machine reports NO MATCH, reproduce it before reading it
as a finding about the run. Re-run the same step on a machine with
more memory, or have another auditor take it. A mismatch that reproduces is
worth reporting as a divergence; one that does not is worth reporting too,
with both hashes, so the failure can be understood. Do not discard either.
