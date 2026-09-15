"""The five outcomes a replay can have, and what each is allowed to do.

Only a match advances the record: no-match, timeout, canceled and inconclusive
change nothing public. Encoding that here rather than in five scattered
branches means the rule is checked in one place, and a sixth outcome cannot be
added without someone deciding what it does.
"""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    """What happened. The wire values are the API's, so this enum is the schema."""

    MATCH = "match"
    """The reproduced digest equals the commitment. The only advancing outcome."""

    NO_MATCH = "no-match"
    """It finished and produced a different digest. A question, not a finding."""

    TIMEOUT = "timeout"
    """It exceeded its deadline without reaching the target step."""

    CANCELED = "canceled"
    """The auditor stopped it, or the process was killed."""

    INCONCLUSIVE = "inconclusive"
    """It ended with no usable verdict: a crash, an OOM, a truncated log."""

    @property
    def advances_record(self) -> bool:
        return self is Outcome.MATCH

    @property
    def uploads_artifact(self) -> bool:
        """Only an accepted match has anything worth handing the next auditor."""
        return self is Outcome.MATCH

    @property
    def headline(self) -> str:
        return {
            Outcome.MATCH: "you independently re-derived the published hash",
            Outcome.NO_MATCH: "your bits differ from the published hash",
            Outcome.TIMEOUT: "the replay ran out of time",
            Outcome.CANCELED: "canceled",
            Outcome.INCONCLUSIVE: "no usable result",
        }[self]

    @property
    def exit_code(self) -> int:
        """0 only for a match. 2 for finished-but-negative, so a script can tell
        "it ran and disagreed" from "it never ran"."""
        return {
            Outcome.MATCH: 0,
            Outcome.NO_MATCH: 2,
            Outcome.TIMEOUT: 2,
            Outcome.CANCELED: 1,
            Outcome.INCONCLUSIVE: 1,
        }[self]


def classify(
    *,
    matched: bool | None,
    exit_code: int | None,
    interrupted: bool = False,
    timed_out: bool = False,
) -> Outcome:
    """Decide the outcome from what the replay actually did.

    Order matters. A run the auditor stopped is canceled even if the log holds
    a verdict from an earlier attempt, and a timeout is a timeout even though
    it also produced no digest -- reporting either as `inconclusive` would lose
    the one fact that explains it.
    """
    if interrupted:
        return Outcome.CANCELED
    if timed_out:
        return Outcome.TIMEOUT
    if matched is True:
        return Outcome.MATCH
    if matched is False:
        return Outcome.NO_MATCH
    # No verdict. Exit 0 with no digest means the process reported success
    # without producing evidence, which is a bug, not a negative result.
    del exit_code
    return Outcome.INCONCLUSIVE


#: Shown after a no-match. Every cause named here is documented and real: the
#: PRD requires the tool to say them out loud so a hardware fault does not read
#: as an accusation against the run.
NO_MATCH_EDUCATION = """This is a question about the run, not a finding against it. Nothing was
published and nothing on the record changed.

Check the provenance first — a result naming a different repop build than the
kit verified different kernels, and that alone explains a mismatch.

If the provenance matches the kit, the likely causes, in order:
  · a memory bit flip, or storage corruption if the machine was deep in swap
  · running outside the pinned environment
  · a genuine reproducibility break, which is rare and is the serious case

One failure is triaged, not published. The record re-runs the step on a cluster
GPU — minutes against the hours it cost you — and comes back to you. If
independent replays on other machines keep landing where yours did, that
becomes an incident report rather than your argument to make.

Your claim lapses on its own. You lose the time and nothing else."""

INCONCLUSIVE_EDUCATION = """The replay ended without a digest, so there is nothing to compare. This is
recorded as inconclusive and changes nothing public.

The log is the place to look. The most common causes are running out of memory
on a machine below the unit's published peak, and a kit whose wheels do not
match the device you asked for."""
