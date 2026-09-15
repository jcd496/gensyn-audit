"""gensyn-audit — the local runner for one auditable-training replay.

The web record coordinates and explains; this tool verifies. It turns a step
number plus a manifest into the exact ``pretrain.cli.audit_replay`` invocation
the MPS runbook prescribes, launches it detached, follows it, and reports the
one thing that matters: whether the state hash you produced equals the hash the
run committed to.

It deliberately reproduces nothing itself. Every bit that is compared is
computed by ``audit_replay`` inside the training repo's own virtualenv, so an
auditor's trust rests on that code and not on this wrapper. ``gensyn-audit plan``
prints the command in full for exactly that reason -- you should be able to
read what this will run before you let it run.
"""

__version__ = "1.0.0"
