"""Terminal output.

One module owns every escape code so `--no-color`, a pipe, and NO_COLOR all
behave the same everywhere. Nothing here decides anything; it only renders.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from typing import Self

_FORCE_PLAIN = False


def set_plain(plain: bool) -> None:
    global _FORCE_PLAIN
    _FORCE_PLAIN = plain


def color_enabled(stream=None) -> bool:
    if _FORCE_PLAIN or os.environ.get("NO_COLOR"):
        return False
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


_CODES = {
    "dim": "2",
    "bold": "1",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
    "brand": "38;2;250;215;209",  # #FAD7D1
}


def paint(text: str, *styles: str) -> str:
    if not styles or not color_enabled():
        return text
    codes = ";".join(_CODES[s] for s in styles if s in _CODES)
    return f"\033[{codes}m{text}\033[0m" if codes else text


def width(default: int = 80) -> int:
    return shutil.get_terminal_size((default, 24)).columns


# ── marks ────────────────────────────────────────────────────────────────────
# Unicode by default, ASCII when the terminal cannot promise UTF-8. A doctor
# report full of mojibake reads as a broken tool.


def _unicode_ok() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


PASS, FAIL, WARN, SKIP = ("✓", "✗", "!", "·") if _unicode_ok() else ("OK", "X", "!", "-")
ARROW = "→" if _unicode_ok() else "->"


def rule(label: str = "") -> str:
    w = min(width(), 78)
    if not label:
        return paint("─" * w if _unicode_ok() else "-" * w, "dim")
    bar = ("─" if _unicode_ok() else "-") * max(0, w - len(label) - 3)
    return paint(f"{label} {bar}", "dim")


def kv(key: str, value: str, note: str = "", key_width: int = 26) -> str:
    """One `key  value  (note)` row, the shape the record's spec lists use."""
    line = f"  {key.ljust(key_width)}{value}"
    if note:
        line += paint(f"   {note}", "dim")
    return line


def echo(text: str = "") -> None:
    print(text)


def warn(text: str) -> None:
    # stdout is block-buffered when piped; without this flush a stderr warning
    # jumps ahead of the lines it is about and reads as being about nothing.
    sys.stdout.flush()
    print(f"{paint(WARN, 'yellow')} {text}", file=sys.stderr)


def fail(text: str, hint: str | None = None) -> None:
    sys.stdout.flush()
    print(f"{paint(FAIL, 'red')} {text}", file=sys.stderr)
    if hint:
        for line in hint.splitlines():
            print(paint(f"    {line}", "dim"), file=sys.stderr)


def head(title: str, sub: str = "") -> None:
    print()
    print(paint(title, "bold"))
    if sub:
        print(paint(sub, "dim"))
    print()


def shell_quote(arg: str) -> str:
    """Quote for display in a copy-pasteable command.

    `shlex.quote` is correct but noisy -- it wraps `gs://a/b` in quotes. Only
    quote what a shell would actually mangle, so `gensyn-audit plan` output reads like
    the runbook it mirrors.
    """
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    safe |= set("@%_-+=:,./~")
    if arg and all(c in safe for c in arg):
        return arg
    return "'" + arg.replace("'", "'\\''") + "'"


def render_command(argv: list[str], indent: str = "  ") -> str:
    """A multi-line, copy-pasteable rendering: one flag (with its value) per line."""
    if not argv:
        return ""
    lines: list[str] = [shell_quote(argv[0])]
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("-") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            lines.append(f"{tok} {shell_quote(argv[i + 1])}")
            i += 2
        else:
            lines.append(shell_quote(tok))
            i += 1
    body = f" \\\n{indent}  ".join(lines)
    return f"{indent}{body}"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ── progress ─────────────────────────────────────────────────────────────────

#: Eighth-blocks, so a bar can advance a fraction of a cell rather than jumping
#: a whole one. On a replay that moves one micro-batch every few minutes, a bar
#: quantised to whole cells looks stuck for a long time.
_EIGHTHS = " ▏▎▍▌▋▊▉█"


def bar(fraction: float | None, cells: int = 30, *, tick: int = 0, estimate: bool = False) -> str:
    """A determinate bar, or a moving pulse when there is no fraction.

    An init unit reports no micro-batches, so a determinate bar would sit at 0%
    for its whole life and read as a hang. A static rule reads the same way, so
    the indeterminate form moves: `tick` advances a short highlight along it,
    which is the only signal that a silent phase is still alive.
    """
    if fraction is None:
        # Wrap the highlight rather than sliding it off the end: the sliding
        # version left one frame in every (cells + span) completely blank, which
        # on a silent phase is the one moment it must not look stopped.
        span, pos = 6, tick % cells
        lit = {(pos + k) % cells for k in range(span)}
        cs = [paint("━", "cyan") if i in lit else paint("─", "dim") for i in range(cells)]
        return "▐" + "".join(cs) + "▌"
    fraction = max(0.0, min(1.0, fraction))
    filled = fraction * cells
    whole = int(filled)
    part = _EIGHTHS[int((filled - whole) * 8)] if whole < cells else ""
    rest = "░" * max(0, cells - whole - len(part))
    # A time estimate is drawn hollow, so a glance separates "this much of the
    # work is done" from "this much of the expected time has passed". They are
    # different claims and only one of them is measured.
    body = ("▒" * whole if estimate else "█" * whole) + part
    return "▐" + paint(body, "blue" if estimate else "cyan") + paint(rest, "dim") + "▌"


class Live:
    """A status block that redraws in place.

    Cursor-up rewriting only where the terminal will honour it. Everywhere else
    -- a pipe, a CI log, `NO_COLOR` -- it prints an occasional plain line
    instead, because a redraw sequence written to a file produces thousands of
    lines of escape codes and no information.
    """

    def __init__(
        self, *, stream=None, min_interval: float = 0.4, plain_interval: float = 60.0
    ) -> None:
        self.stream = stream or sys.stdout
        self.tty = color_enabled(self.stream)
        self.min_interval = min_interval
        self.plain_interval = plain_interval
        self._drawn = 0
        self._last = 0.0

    def update(self, lines: list[str], *, force: bool = False) -> None:
        import time

        now = time.monotonic()
        gap = self.min_interval if self.tty else self.plain_interval
        if not force and (now - self._last) < gap:
            return
        self._last = now

        if not self.tty:
            # One line, no cursor tricks: readable in a log, and rare enough
            # not to fill one.
            if lines:
                print(lines[0].strip(), file=self.stream, flush=True)
            return

        out = []
        if self._drawn:
            out.append(f"\033[{self._drawn}A")
        for line in lines:
            out.append("\033[2K" + line + "\n")
        self.stream.write("".join(out))
        self.stream.flush()
        self._drawn = len(lines)

    def clear(self) -> None:
        """Erase the block, so the final result is not printed under a stale bar."""
        if self.tty and self._drawn:
            self.stream.write(
                f"\033[{self._drawn}A" + "\033[2K\n" * self._drawn + f"\033[{self._drawn}A"
            )
            self.stream.flush()
        self._drawn = 0


class Activity:
    """A pulse for a blocking step that prints nothing of its own.

    Packing an 18 GB hand-off, unpacking one, reconstructing its hash and
    digesting it each take minutes inside a subprocess that says nothing until
    it is done. This keeps the indeterminate bar moving while the caller blocks
    and leaves one line saying what happened and how long it took.

        with Activity("packing the hand-off"):
            convert.pack(...)
    """

    def __init__(self, label: str, *, detail: str = "", interval: float = 0.25) -> None:
        self.label = label
        self.detail = detail
        self.interval = interval
        self._live = Live()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def _lines(self, tick: int) -> list[str]:
        import time

        elapsed = human_duration(time.monotonic() - self._started)
        label = paint(f"{self.label:<22}", "cyan")
        where = paint(self.detail, "dim") if self.detail else ""
        return [
            f"  {label} {where}".rstrip(),
            f"  {bar(None, tick=tick)}       {paint(elapsed, 'dim')}",
        ]

    def __enter__(self) -> Self:
        import time

        self._started = time.monotonic()
        self._stop.clear()

        def pulse() -> None:
            tick = 0
            while not self._stop.wait(self.interval):
                tick += 1
                self._live.update(self._lines(tick))

        self._live.update(self._lines(0), force=True)
        self._thread = threading.Thread(target=pulse, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import time

        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._live.clear()
        took = human_duration(time.monotonic() - self._started)
        mark = paint(PASS, "green") if exc_type is None else paint(FAIL, "red")
        verb = "" if exc_type is None else " failed"
        echo(f"  {mark} {self.label}{verb}  {paint(took, 'dim')}")
