"""The memory check says what a machine is in for, not just whether it fits.

OPEN-1B, first week: a 24 GB MacBook Pro passed preflight, then spent ~18 hours
in swap on a step a 48 GB machine did in 6, and two of three such replays
reported a hash that matched nothing -- differently each time. The check let
that machine through without a word.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gensyn_audit import doctor


def _interval():
    return SimpleNamespace(is_init=False, mps_peak_rss_gb=None)


@pytest.mark.parametrize(
    "have, status", [(16, doctor.FAIL), (24, doctor.WARN), (36, doctor.WARN), (48, doctor.PASS)]
)
def test_the_memory_verdict_by_machine(monkeypatch, have, status):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: float(have))
    assert doctor._check_memory(_interval(), "mps").status == status


def test_a_swapping_machine_is_told_the_hours_and_what_to_do_with_a_mismatch(monkeypatch):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: 24.0)
    check = doctor._check_memory(_interval(), "mps")
    assert not check.blocking, "possible on ordinary hardware is the point; warn, do not refuse"
    assert "18 hours on a 24 GB" in check.fix
    assert "6 hours on a 48 GB" in check.fix
    assert "under an hour on an H100" in check.fix
    assert "NO MATCH" in check.fix
    assert "reproduce" in check.fix
    # Two unrepeatable mismatches were observed; what caused them was not.
    # The warning must not diagnose them, and must not tell an auditor that a
    # mismatch from a small machine says nothing.
    assert "corruption" not in check.fix
    assert "not evidence" not in check.fix


def test_an_init_unit_is_not_warned_about_swap(monkeypatch):
    monkeypatch.setattr(doctor, "_memory_gb", lambda: 24.0)
    unit = SimpleNamespace(is_init=True, mps_peak_rss_gb=5.8)
    assert doctor._check_memory(unit, "mps").status == doctor.PASS
