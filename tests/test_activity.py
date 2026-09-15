"""The pulse that stands in for a silent subprocess."""

from __future__ import annotations

import time

import pytest

from gensyn_audit.errors import AuditError
from gensyn_audit.ui import Activity


def test_an_activity_leaves_one_line_saying_what_happened(capsys):
    with Activity("packing the hand-off", detail="minutes at 18 GB"):
        time.sleep(0.05)
    out = capsys.readouterr().out
    assert "packing the hand-off" in out
    assert "failed" not in out


def test_a_failing_activity_says_so_and_lets_the_error_through(capsys):
    with pytest.raises(AuditError), Activity("unpacking the hand-off"):
        raise AuditError("the converter refused")
    assert "unpacking the hand-off failed" in capsys.readouterr().out


def test_a_pipe_gets_plain_lines_not_cursor_movement(capsys):
    """The supervised child's stdout is a log file. Redraw sequences there
    would be thousands of lines of escape codes."""
    with Activity("digesting the hand-off"):
        time.sleep(0.05)
    assert "\033[" not in capsys.readouterr().out
