"""The Gensyn mark, and the header it sits in.

The art below is the canonical symbol from ``brand.gensyn.ai`` (logo-symbol,
``viewBox="0 0 306 306"``) rasterised onto a 32x16 terminal grid. That size is
not arbitrary: the symbol is a stepped octagon whose walls are 36.5 units on a
300-unit field and whose faces are 81, so at 32 columns a wall lands on exactly
4 cells and a face on 8, and nothing rounds. Smaller grids collapse the thick
walls into a thin diamond -- a different shape, and not the brand's.

Terminal cells are about twice as tall as they are wide, so 32x16 reads square.

The mark is decoration and must never be load-bearing: it degrades to plain
text when output is not a terminal, when NO_COLOR is set, or when the terminal
is too narrow to hold it.
"""

from __future__ import annotations

from . import __version__
from .ui import PASS, WARN, color_enabled, paint, width

#: 32x16, from the canonical symbol. Do not hand-edit -- regenerate from the
#: published SVG if the brand asset ever changes.
MARK: tuple[str, ...] = (
    "            ████████",
    "            ████████",
    "        ████        ████",
    "        ████        ████",
    "    ████                ████",
    "    ████                ████",
    "████                        ████",
    "████                        ████",
    "████                        ████",
    "████                        ████",
    "    ████                ████",
    "    ████                ████",
    "        ████        ████",
    "        ████        ████",
    "            ████████",
    "            ████████",
)

#: The one-glyph fallback. U+25C8 reads as the mark's silhouette at 1x1.
GLYPH = "\u25c8"

WORDMARK = "gensyn audit"

#: Below this the mark would wrap, which looks worse than not drawing it.
MIN_WIDTH = 38


def mark_lines(indent: str = "  ") -> list[str]:
    return [indent + line for line in MARK]


def banner(*, force: bool | None = None) -> list[str]:
    """The header block: art where there is room, one line otherwise."""
    show = force if force is not None else (color_enabled() and width() >= MIN_WIDTH)
    if not show:
        return [f"  {GLYPH} {WORDMARK} {__version__}"]
    return [
        "",
        *[paint(line, "brand") for line in mark_lines()],
        "",
        f"  {paint(WORDMARK, 'bold')} {paint(__version__, 'dim')}",
    ]


def completion_banner(*, plain: bool = False) -> list[str]:
    """A matched audit submitted with its hand-off, not server acceptance."""

    def style(text: str, *styles: str) -> str:
        return text if plain else paint(text, *styles)

    art = (
        [*[paint(line, "brand") for line in mark_lines()], ""]
        if not plain and color_enabled() and width() >= MIN_WIDTH
        else []
    )
    return [
        "",
        *art,
        style("  AUDIT MATCHED. CONTRIBUTION SUBMITTED.", "green", "bold"),
        "",
        "  You independently reproduced a published training-state hash.",
        "  Thank you for helping audit Open 1B.",
        "",
        style(f"  {PASS} Audit matched", "green"),
        style(f"  {PASS} Submission recorded", "green"),
        style(f"  {PASS} Checkpoint and sidecar uploaded", "green"),
        style(f"  {WARN} Submitted for server verification.", "yellow"),
        "    Check the receipt for the latest status.",
        "",
    ]
