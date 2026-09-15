"""The logo, the bar, and the live block.

Chrome, but chrome that is on screen for most of a day and has to behave when
it is not on a screen at all.
"""

from __future__ import annotations

import io
import re

from gensyn_audit import brand, cli, progress, ui

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    return ANSI.sub("", text)


# ── the mark ─────────────────────────────────────────────────────────────────


def test_the_mark_is_the_published_geometry():
    """32x16, because that is where the octagon's walls (36.5 of 300 units) and
    faces (81) land on whole cells. A smaller grid collapses the thick walls
    into a thin diamond, which is a different shape."""
    assert len(brand.MARK) == 16
    assert max(len(r) for r in brand.MARK) == 32

    filled = [[c == "█" for c in row.ljust(32)] for row in brand.MARK]
    # Hollow: the middle is empty.
    assert not any(filled[8][12:20])
    # Symmetric left-to-right and top-to-bottom, as the asset is.
    for row in filled:
        assert row == row[::-1]
    assert filled == filled[::-1]
    # The top face is 8 cells of wall, the side wall 4.
    assert sum(filled[0]) == 8
    assert sum(filled[8]) == 8  # 4 cells each side


def test_the_banner_degrades_to_one_line_off_a_terminal():
    """A pipe or a CI log gets text, not sixteen rows of block characters."""
    assert brand.banner(force=False) == [f"  {brand.GLYPH} gensyn audit {brand.__version__}"]
    assert len(brand.banner(force=True)) > 16


def test_completion_uses_the_brand_color_only_on_a_wide_color_terminal(monkeypatch):
    monkeypatch.setattr(ui, "_FORCE_PLAIN", False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(brand, "width", lambda: 80)
    colored = "\n".join(brand.completion_banner())
    assert "\x1b[32m" in colored and "\x1b[33m" in colored
    for banner in (colored, "\n".join(brand.banner())):
        for line in brand.mark_lines():
            assert f"\x1b[38;2;250;215;209m{line}\x1b[0m" in banner
    assert "Submitted for server verification." in colored
    assert "Check the receipt for the latest status." in colored

    monkeypatch.setattr(brand, "width", lambda: 30)
    narrow = "\n".join(brand.completion_banner())
    assert "█" not in narrow and "AUDIT MATCHED." in narrow
    monkeypatch.setattr(brand, "width", lambda: 80)
    monkeypatch.setenv("NO_COLOR", "1")
    plain = "\n".join(brand.completion_banner())
    assert "█" not in plain and "\x1b" not in plain
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setattr(ui, "_FORCE_PLAIN", True)
    assert "\n".join(brand.completion_banner()) == plain
    monkeypatch.setattr(ui, "_FORCE_PLAIN", False)
    monkeypatch.setattr(ui.sys.stdout, "isatty", lambda: False)
    assert "\n".join(brand.completion_banner()) == plain


# ── the bar ──────────────────────────────────────────────────────────────────


def test_the_bar_fills_proportionally():
    ui.set_plain(True)
    try:
        assert _plain(ui.bar(0.0, 10)).count("█") == 0
        assert _plain(ui.bar(1.0, 10)).count("█") == 10
        assert _plain(ui.bar(0.5, 10)).count("█") == 5
    finally:
        ui.set_plain(False)


def test_the_bar_advances_by_eighths():
    """A replay moves one micro-batch every few minutes; a bar quantised to
    whole cells would look stuck for most of that."""
    ui.set_plain(True)
    try:
        a, b = _plain(ui.bar(0.50, 10)), _plain(ui.bar(0.52, 10))
        assert a != b, "a 2% move must be visible"
    finally:
        ui.set_plain(False)


def test_out_of_range_fractions_are_clamped():
    ui.set_plain(True)
    try:
        assert _plain(ui.bar(-1.0, 10)).count("█") == 0
        assert _plain(ui.bar(9.0, 10)).count("█") == 10
    finally:
        ui.set_plain(False)


def test_the_indeterminate_bar_moves():
    """An init unit reports no micro-batches and spends ~30s silent. A static
    rule there is indistinguishable from a hang."""
    ui.set_plain(True)
    try:
        frames = {_plain(ui.bar(None, 20, tick=t)) for t in range(0, 20, 3)}
        assert len(frames) > 3, "the pulse must actually travel"
        assert all(len(f) == 22 for f in frames), "width must not jitter"
    finally:
        ui.set_plain(False)


# ── the live block ───────────────────────────────────────────────────────────


def test_live_writes_no_escape_codes_when_not_a_terminal():
    """Cursor-up sequences in a log file are thousands of lines of noise."""
    buf = io.StringIO()
    live = ui.Live(stream=buf, plain_interval=0.0)
    for _ in range(3):
        live.update(["  replaying   50%", "  second line"], force=True)
    out = buf.getvalue()
    assert "\x1b[" not in out
    assert "second line" not in out, "only a single summary line off-TTY"


def test_live_clear_is_a_noop_off_a_terminal():
    buf = io.StringIO()
    live = ui.Live(stream=buf)
    live.clear()
    assert buf.getvalue() == ""


def test_live_throttles_redraws():
    buf = io.StringIO()
    live = ui.Live(stream=buf, plain_interval=1000.0)
    live.update(["first"], force=True)
    live.update(["second"])
    assert "second" not in buf.getvalue()


# ── phases ───────────────────────────────────────────────────────────────────


def test_an_init_unit_reports_its_real_phases():
    """It used to read `starting up` for its entire 40-second life."""

    def phase_after(msg: str) -> str:
        return progress.parse(f"2026-09-07 10:00:00,000 INFO pretrain.audit :: {msg}\n").phase

    assert phase_after("config-only init audit: seed=42 config=1b_repop_run3") == "building"
    assert phase_after("param groups: decay=38 (234.7M), no_decay=49") == "generating"
    assert phase_after("replaying steps 100 → 200") == "replaying"


def test_every_phase_has_a_label():
    """An unlabelled phase would surface its internal name to the auditor."""
    for phase in (
        "starting",
        "building",
        "fetching",
        "loading",
        "generating",
        "hashing",
        "replaying",
        "done",
        "failed",
    ):
        assert phase in cli._PHASE_LABEL, phase


def test_the_phase_column_is_padded_on_visible_width():
    """Regression: padding a colour-painted string counts the escape bytes, so
    the column silently came out ~9 characters short."""
    assert cli._PHASE_WIDTH >= max(len(v) for v in cli._PHASE_LABEL.values())

    prog = progress.Progress(phase="generating")
    lines = cli._status_block(prog, 0.0, "1b_repop_run3 init")
    visible = _plain(lines[0])
    assert visible.startswith("  regenerating the initial state")
    assert "1b_repop_run3 init" in visible
    # The counts column must sit at the same place for a short label too.
    short = _plain(cli._status_block(progress.Progress(phase="done"), 0.0, "x")[0])
    assert visible.index("1b_repop") == short.index("x")


# ── the command's name ───────────────────────────────────────────────────────


def test_the_entry_point_does_not_collide_with_a_system_binary():
    """macOS ships /usr/sbin/audit. A console script called `audit` shadows it
    when a venv is active and is shadowed BY it when one is not, which is the
    worst of both: the same word runs different programs depending on shell
    state. Pinned here because the name is easy to "simplify" back."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    assert "audit" not in scripts
    assert scripts["gensyn-audit"] == "gensyn_audit.cli:main"


def test_help_and_hints_use_the_real_command_name():
    """A hint telling someone to run `audit run` sends them to Apple's auditd."""
    parser = cli.build_parser()
    assert parser.prog == "gensyn-audit"

    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "gensyn_audit"
    import re

    stale = re.compile(r"(?<![-\w])audit (run|units|doctor|plan|install|status|logs)\b")
    offenders = [
        f"{p.name}:{n}"
        for p in src.glob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if stale.search(line)
    ]
    assert not offenders, "these still tell the user to run `audit …`: " + ", ".join(offenders)


# ── progress when there is no fraction to report ─────────────────────────────


def _bar_line(prog, elapsed_started, expected=None, label="1b_repop_run3 init"):
    return _plain(cli._status_block(prog, elapsed_started, label, expected)[1])


def test_an_init_unit_gets_a_time_estimate_not_a_pulse(monkeypatch):
    """1b_repop_run3 emits nothing for 36 seconds between building the model
    and the digest — measured. There is no fraction to report, so elapsed
    against the trajectory's published time is the only honest signal."""
    import time

    started = time.monotonic() - 15  # 15s into a ~30s unit
    line = _bar_line(progress.Progress(phase="generating"), started, expected=30.0)
    import re as _re

    pct = int(_re.search(r"~\s*(\d+)%", line).group(1))
    assert 48 <= pct <= 52, line
    assert "▒" in line, "an estimate is drawn hollow, not solid"
    assert "expected" in line


def test_the_estimate_holds_short_of_full_when_it_overruns():
    """It is a guess against another machine's measurement. A bar sitting at
    100% while the work continues is a worse lie than one sitting at 95%."""
    import time

    started = time.monotonic() - 300  # ten times the expected
    line = _bar_line(progress.Progress(phase="generating"), started, expected=30.0)
    assert "~95%" in line
    assert "100%" not in line
    assert "over the" in line


def test_a_real_fraction_is_never_drawn_as_an_estimate():
    """A measured micro-batch count must look different from a time guess."""
    prog = progress.Progress(phase="replaying", microbatches_done=144, microbatches_total=288)
    line = _bar_line(prog, 0.0, expected=30.0)
    assert "█" in line and "▒" not in line
    assert "~" not in line.split("%")[0], "no tilde on a measured percentage"


def test_completion_is_solid_and_full():
    """Regression: the final frame fell back to the indeterminate pulse, which
    read as the bar going backwards at the moment it finished."""
    line = _bar_line(progress.Progress(phase="done"), 0.0, expected=30.0)
    assert "100%" in line
    assert "━" not in line, "must not revert to the pulse"


def test_no_expected_time_still_gives_a_pulse():
    """A unit whose cost the trajectory never published (1b_repop_v2) has
    nothing to estimate against."""
    line = _bar_line(progress.Progress(phase="generating"), 0.0, expected=None)
    assert "━" in line and "%" not in line


def test_every_command_a_hint_names_actually_exists():
    """Regression: the kit rewrite dropped `stop` but left a hint telling
    people to run it, so the advice printed when a replay was already running
    pointed at a command that did not exist."""
    import re
    from pathlib import Path

    subcommands = set(cli.build_parser()._subparsers._group_actions[0].choices)
    src = Path(__file__).resolve().parents[1] / "src" / "gensyn_audit"
    named = re.compile(r"gensyn-audit ([a-z][a-z-]*)")

    missing = {
        f"{p.name}:{n} -> {m}"
        for p in src.glob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        for m in named.findall(line)
        if m not in subcommands
    }
    assert not missing, "hints name commands that do not exist: " + ", ".join(sorted(missing))


def test_the_pulse_is_never_invisible():
    """It used to slide off the end: one frame in every 36 rendered a bar with
    no highlight at all, which on a silent phase is the single moment it must
    not look stopped. (It also made the pulse test flaky.)"""
    ui.set_plain(True)
    try:
        for tick in range(200):
            drawn = _plain(ui.bar(None, 30, tick=tick))
            assert drawn.count("━") == 6, f"tick {tick}: {drawn}"
            assert len(drawn) == 32
    finally:
        ui.set_plain(False)


# ── status: the live block ───────────────────────────────────────────────────


def test_an_interval_is_not_told_it_is_regenerating_an_initial_state():
    """`param groups` is the last line before an init unit's long silent
    generate-and-hash, which is what that phase label describes. An interval
    logs the same line on its way into loading an 18 GB checkpoint."""
    import time

    from gensyn_audit import cli
    from gensyn_audit.progress import Progress

    prog = Progress(phase="generating")
    started = time.monotonic()

    init = "\n".join(cli._status_block(prog, started, "u", is_init=True))
    assert "regenerating the initial state" in init

    interval = "\n".join(cli._status_block(prog, started, "u", is_init=False))
    assert "regenerating" not in interval
    assert "optimizer" in interval


def test_elapsed_origin_matches_a_start_recorded_by_another_process():
    """A detached replay records an ISO start; `_status_block` measures against
    time.monotonic(), which is only comparable within one process."""
    import datetime as dt
    import time

    from gensyn_audit import cli

    began = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=600)
    origin = cli._elapsed_origin(began.isoformat())
    assert 595 <= (time.monotonic() - origin) <= 605


def test_an_unparsable_start_does_not_crash_status():
    import time

    from gensyn_audit import cli

    for bad in ("", "not-a-date", None):
        assert (time.monotonic() - cli._elapsed_origin(bad)) < 1.0
