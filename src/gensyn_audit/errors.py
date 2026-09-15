"""Errors that carry a fix, not just a complaint."""

from __future__ import annotations


class AuditError(Exception):
    """A failure the auditor can act on.

    ``hint`` is printed under the message as an indented remedy. Everything
    raised out of this package should be one of these: an unhandled traceback
    in a tool people run for eighteen hours is a bug report we cannot action.
    """

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


class RunStateError(AuditError):
    """The workdir does not hold the run state a command needs."""
