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

## Memory and swap

Measured in the first week of Open 1B audits, a step takes roughly:

| Machine                | Time per step   |
|------------------------|-----------------|
| 24 GB MacBook Pro      | about 18 hours  |
| 48 GB MacBook Pro      | about 6 hours   |
| H100                   | under an hour   |

There is no minimum spec. The point is that an audit is doable on ordinary
hardware. Below about 40 GB of unified memory the replay lives in swap, so
expect the long form. `doctor` warns about this rather than refusing.

In that same week, two of three replays on 24 GB machines reported a hash that
matched nothing, a different one each time. What caused those two divergences
was not established. Nothing was measured that would separate a memory fault
from nondeterminism, a software defect or another hardware problem, so this
guide does not name a cause.

What to do with a NO MATCH from a heavily swapping machine: reproduce it before
reading it as a finding about the run. Re-run the same step on a machine with
more memory, or have another auditor take it. A mismatch that reproduces is
worth reporting as a divergence; one that does not is worth reporting too,
with both hashes, so the failure can be understood. Do not discard either.
