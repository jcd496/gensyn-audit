"""Reading a replay's log.

`audit_replay` is a training loop, not a service: it reports through Python
logging and two tqdm bars, and its verdict is one line plus an exit code. This
module is the only thing that knows those shapes, so `status`, `logs` and
`submit` all agree on what the run is doing.

Two details worth stating, because getting them wrong produces a confident
wrong answer:

* The `MATCH=` line carries digests **truncated to 16 characters**. The full
  value comes from the JSON the process prints on success, or from the
  `AUDIT FAILED:` line on failure. Never report the truncated one as the hash.
* tqdm writes with carriage returns, so the newest bar is the last `\\r`-
  separated field of the last line -- not the last line.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: `2026-08-27 10:11:12,345 INFO pretrain.cli.audit_replay :: message`
_LOG_LINE = re.compile(
    r"^(?P<time>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\s+(?P<level>\w+)\s+\S+\s+::\s+(?P<msg>.*)$"
)

_REPLAYING = re.compile(r"replaying steps (\d+) . (\d+)")
_MATCH = re.compile(
    r"state_hash=(?P<got>[0-9a-f]+) expected=(?P<want>[0-9a-f]+) MATCH=(?P<match>True|False)"
)
_FAILED = re.compile(r"AUDIT FAILED: state_hash (?P<got>[0-9a-f]{64}) != (?P<want>[0-9a-f]{64})")
_MEMLOG = re.compile(
    r"MEMLOG\[(?P<tag>[^\]]*)\] host_RSS=(?P<rss>[\d.]+)GB"
    r"(?: mps_alloc=(?P<alloc>[\d.]+)GB mps_driver=(?P<driver>[\d.]+)GB)?"
)
_SAVED = re.compile(r"saved chained-audit checkpoint . (?P<dir>\S+)")
#: `  step 801 microbatches:  42%|####  | 121/288 [1:23:45<3:12:01,  1.2s/mb]`
_BAR = re.compile(
    r"(?P<desc>[\w ]*microbatches):\s*(?P<pct>\d+)%\|[^|]*\|\s*(?P<done>\d+)/(?P<total>\d+)"
    r"(?:\s*\[(?P<elapsed>[\d:]+)<(?P<eta>[\d:?]+)[,\]])?"
)
_HALT = re.compile(r"spike protocol halt at step (\d+)")


@dataclass
class Progress:
    """What the log says, as of the last read."""

    phase: str = "starting"
    """One of: starting, loading, fetching, replaying, hashing, done, failed."""

    from_step: int | None = None
    until_step: int | None = None

    microbatches_done: int | None = None
    microbatches_total: int | None = None
    bar_eta: str | None = None
    bar_elapsed: str | None = None

    host_rss_gb: float | None = None
    mps_alloc_gb: float | None = None
    mps_driver_gb: float | None = None

    match: bool | None = None
    state_hash: str | None = None
    """Full 64-hex when known; None while only the truncated log line has been seen."""
    expected_hash: str | None = None
    state_hash_short: str | None = None
    expected_hash_short: str | None = None

    mode: str | None = None
    """``init`` for a config-only init audit; absent for an interval replay."""

    repop_commit: str | None = None
    repop_backends: tuple[str, ...] = ()
    device: str | None = None
    """Kernel-build provenance, straight from the result. Never inferred."""

    rank0_losses: list[dict] = field(default_factory=list)
    """One record per replayed step: the values the record's loss gate runs on."""

    saved_checkpoint: str | None = None
    halted_at_step: int | None = None
    last_message: str = ""
    last_timestamp: datetime | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def fraction(self) -> float | None:
        if not self.microbatches_total:
            return None
        return min(1.0, (self.microbatches_done or 0) / self.microbatches_total)

    @property
    def finished(self) -> bool:
        return self.phase in ("done", "failed")

    @property
    def reproduced(self) -> str | None:
        """The digest the replay produced, full when the result block gave it.

        The full digest reaches us only from the trailing JSON audit_replay
        prints on the success path. A mismatch raises before that block, so on
        the one outcome where the number is worth reading, all that survives is
        the 16-hex prefix in the `state_hash=` log line. Reporting the prefix
        beats reporting nothing: the first mismatch a tester hit was escalated
        to a maintainer purely because the CLI showed `?` for a value that was
        sitting in the log the whole time.
        """
        return self.state_hash or self.state_hash_short

    @property
    def reproduced_is_truncated(self) -> bool:
        """True when `reproduced` is the log's 16-hex prefix, not the digest."""
        return not self.state_hash and bool(self.state_hash_short)


#: Ordered least- to most-advanced: a later match wins, so the phase only ever
#: moves forward. The init markers come from lines audit_replay already logs --
#: `param groups` is the last thing it says before spending most of its runtime
#: generating and hashing the initial state in silence.
_PHASE_HINTS = (
    ("building", ("config-only init audit", "parallelize applies", "loss: fused")),
    ("fetching", ("fetch", "downloading", "shards")),
    ("loading", ("auditing ", "repop_env", "loading", "offload ON")),
    ("generating", ("param groups",)),
    ("hashing", ("state_hash=",)),
)


def parse(text: str) -> Progress:
    """Parse a whole log. Cheap enough to redo on every `status` poll."""
    p = Progress()

    # tqdm's carriage returns mean the interesting bar is not on its own line.
    for chunk in reversed(text.replace("\r", "\n").splitlines()):
        if bar := _BAR.search(chunk):
            p.microbatches_done = int(bar["done"])
            p.microbatches_total = int(bar["total"])
            p.bar_eta = bar["eta"]
            p.bar_elapsed = bar["elapsed"]
            break

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if failed := _FAILED.search(line):
            # The only place both full digests appear together on the sad path.
            p.state_hash, p.expected_hash = failed["got"], failed["want"]
            p.match, p.phase = False, "failed"
            continue

        entry = _LOG_LINE.match(line)
        msg = entry["msg"] if entry else line
        if entry:
            try:
                # Naive on purpose: audit_replay's asctime carries no offset,
                # so this is local wall time and pretending otherwise would be
                # a lie. Nothing subtracts it from a UTC value -- elapsed is
                # computed from epoch times, not from these.
                p.last_timestamp = datetime.strptime(  # noqa: DTZ007
                    entry["time"], "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                pass
            if entry["level"] in ("ERROR", "CRITICAL"):
                p.errors.append(msg)
            if entry["level"] != "WARNING" or "MEMLOG" in msg:
                p.last_message = msg

        if m := _REPLAYING.search(msg):
            p.from_step, p.until_step = int(m[1]), int(m[2])
            p.phase = "replaying"
        if m := _MEMLOG.search(msg):
            p.host_rss_gb = float(m["rss"])
            if m["alloc"]:
                p.mps_alloc_gb, p.mps_driver_gb = float(m["alloc"]), float(m["driver"])
        if m := _MATCH.search(msg):
            p.state_hash_short, p.expected_hash_short = m["got"], m["want"]
            p.match = m["match"] == "True"
            p.phase = "done" if p.match else "failed"
        if m := _SAVED.search(msg):
            p.saved_checkpoint = m["dir"]
        if m := _HALT.search(msg):
            p.halted_at_step = int(m[1])
        if p.phase in ("starting", "loading", "fetching", "building", "generating"):
            for phase, needles in _PHASE_HINTS:
                if any(n in msg for n in needles):
                    p.phase = phase

    # The success path prints the result object; it is the only source of the
    # full reproduced digest when the hashes matched.
    if result := _trailing_json(text):
        p.state_hash = result.get("state_hash") or p.state_hash
        p.expected_hash = result.get("expected") or p.expected_hash
        if "match" in result:
            p.match = bool(result["match"])
            p.phase = "done" if p.match else "failed"
        elif p.match is None and p.state_hash:
            p.phase = "done"
        p.saved_checkpoint = result.get("saved_checkpoint") or p.saved_checkpoint
        p.mode = result.get("mode") or p.mode
        p.device = result.get("device") or p.device
        if isinstance(result.get("repop"), dict):
            p.repop_commit = result["repop"].get("commit")
            p.repop_backends = tuple(result["repop"].get("backends") or ())
        if isinstance(result.get("rank0_losses"), list):
            p.rank0_losses = result["rank0_losses"]

    return p


def _trailing_json(text: str) -> dict | None:
    """The `json.dumps(res, indent=2)` block audit_replay prints when it wins.

    Scanning backwards for a line that is exactly `{` is enough: the block is
    indent-2 pretty-printed at column zero, and nothing else in the log is.
    """
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].rstrip() != "{":
            continue
        for j in range(i + 1, len(lines)):
            if lines[j].rstrip() == "}":
                try:
                    doc = json.loads("\n".join(lines[i : j + 1]))
                except json.JSONDecodeError:
                    break
                return doc if isinstance(doc, dict) and "state_hash" in doc else None
            if lines[j].startswith("{"):
                break
    return None


def read_loss_log(path: Path) -> list[dict]:
    """The ``--loss-log`` file: per-step CE and z-loss, live.

    audit_replay rewrites it atomically after every step (temp + os.replace),
    so a read during a replay always sees a whole document -- which is what
    makes it safe to poll while a step is running, unlike the result object
    that only exists once the process exits.
    """
    if not path.is_file():
        return []
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    records = doc.get("records")
    return records if isinstance(records, list) else []


def parse_file(path: Path) -> Progress:
    if not path.is_file():
        return Progress()
    return parse(path.read_text(errors="replace"))


#: Enough to hold the trailing result object, the last bar redraws and a few
#: hundred log lines. Re-reading a whole day's log every 400ms would not.
TAIL_BYTES = 128 * 1024


def parse_tail(path: Path, limit: int = TAIL_BYTES) -> Progress:
    """Parse only the end of the log, for live status during a long replay."""
    if not path.is_file():
        return Progress()
    with open(path, "rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        fh.seek(max(0, size - limit))
        return parse(fh.read().decode("utf-8", errors="replace"))
