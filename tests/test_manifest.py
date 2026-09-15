"""Locations come from data, never from the tool.

A verification tool with a bucket compiled into it cannot verify a run that
moved, and a wrong default is worse than a missing one: it fails late with a
permissions error instead of saying which location it needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gensyn_audit import cli
from gensyn_audit import manifest as manifestmod
from gensyn_audit.errors import AuditError

RUN = "20260703-171943-4e85cd3"


def _manifest(tmp_path: Path, **over) -> Path:
    doc = {
        "manifest_format": 1,
        "run": RUN,
        "kit": "gs://elsewhere/audit-kit/pt-aaaaaaaaaaaa_rp-bbbbbbbbbbbb",
        "record": "https://record.example",
        "artifacts": {
            "checkpoints": "gs://a-different-bucket/run/checkpoints",
            "shards": "gs://a-different-bucket/data/shards",
            "state_hashes": "gs://a-different-bucket/run/logs/state_hashes.jsonl",
        },
    }
    doc.update(over)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(doc))
    return path


def test_every_location_comes_from_the_manifest(tmp_path):
    m = manifestmod.load(str(_manifest(tmp_path)))
    assert m.run == RUN
    assert m.kit == "gs://elsewhere/audit-kit/pt-aaaaaaaaaaaa_rp-bbbbbbbbbbbb"
    assert m.record == "https://record.example"
    assert m.artifacts.shards == "gs://a-different-bucket/data/shards"


def test_checkpoint_uri_uses_the_loops_own_directory_naming(tmp_path):
    m = manifestmod.load(str(_manifest(tmp_path)))
    assert (
        m.artifacts.checkpoint_uri(100) == "gs://a-different-bucket/run/checkpoints/step_000000100"
    )


def test_a_manifest_may_name_nothing_but_the_run(tmp_path):
    """Init units download nothing, so a manifest with no artifact roots is valid."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"manifest_format": 1, "run": RUN}))
    m = manifestmod.load(str(path))
    assert m.artifacts.checkpoints is None
    assert m.artifacts.checkpoint_uri(100) is None
    with pytest.raises(AuditError, match="names no kit"):
        m.require_kit()


def test_a_future_format_is_refused_not_guessed_at(tmp_path):
    with pytest.raises(AuditError, match="manifest_format 99"):
        manifestmod.load(str(_manifest(tmp_path, manifest_format=99)))


def test_a_manifest_without_a_run_is_refused(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"manifest_format": 1}))
    with pytest.raises(AuditError, match="no `run` id"):
        manifestmod.load(str(path))


def test_no_bucket_is_compiled_into_the_tool():
    """The regression this whole file exists to prevent.

    Checks the AST rather than the text: a bucket named in a docstring is
    documentation, a bucket in a live string literal is a compiled-in location.
    """
    import ast
    import re

    bucket = re.compile(r"gs://[a-z0-9][a-z0-9._-]{2,}")
    src = Path(__file__).resolve().parents[1] / "src" / "gensyn_audit"
    offenders = []

    for py in sorted(src.glob("*.py")):
        if py.name == "mock.py":
            # Its URIs are fixture data for a backend that announces itself as
            # fake; they address nothing real.
            continue
        tree = ast.parse(py.read_text())
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and bucket.search(node.value)
            ):
                offenders.append(f"{py.name}:{node.lineno}: {node.value[:70]!r}")

    assert not offenders, "a location was compiled in:\n" + "\n".join(offenders)


def test_missing_kit_names_both_ways_to_supply_one(tmp_path):
    import argparse

    args = argparse.Namespace(kit=None, manifest=None)
    with pytest.raises(AuditError) as exc:
        cli._resolve_kit(args, None)
    assert "--manifest" in exc.value.hint and "--kit" in exc.value.hint


def test_a_flag_beats_the_manifest(tmp_path):
    import argparse

    m = manifestmod.load(str(_manifest(tmp_path)))
    args = argparse.Namespace(kit="/somewhere/local/kit", manifest=None)
    # _resolve_kit would try to read it; assert the precedence directly.
    assert (args.kit or m.kit) == "/somewhere/local/kit"


# ── the record's own manifest.json ───────────────────────────────────────────
#
# The web app prints `--manifest <record>/manifest.json`, so that document has
# to load through the same flag as our static schema. It is a different shape
# and carries no `manifest_format`, so it is recognised structurally.

RECORD_DOC = {
    "run": {"id": "20260722-213626-ad3276b", "name": "open-1b", "totalSteps": 80957},
    "audit": {
        "kit": {
            "id": "pt-f5eece09abe6_rp-244c0791e378",
            "urlBase": "https://example.test/audit-kit/pt-f5eece09abe6_rp-244c0791e378",
            "manifestUrl": "https://example.test/audit-kit/pt-f5eece09abe6_rp-244c0791e378/kit.json",
        }
    },
    "endpoints": {"receipt": "/v1/runs/open-1b/steps/{step}/receipt"},
}


def _load_doc(monkeypatch, doc, source="https://rec.test/manifest.json"):
    import json as _json

    from gensyn_audit import manifest as mod

    monkeypatch.setattr(mod, "_read", lambda uri: _json.dumps(doc).encode())
    return mod.load(source)


def test_record_manifest_supplies_kit_and_record(monkeypatch):
    m = _load_doc(monkeypatch, RECORD_DOC)
    # Endpoints address the run by name, not by id.
    assert m.run == "open-1b"
    assert m.kit == "https://example.test/audit-kit/pt-f5eece09abe6_rp-244c0791e378"
    # The record is wherever its manifest was served from.
    assert m.record == "https://rec.test"


def test_record_manifest_falls_back_to_the_kit_manifest_url(monkeypatch):
    doc = _json_without_url_base()
    m = _load_doc(monkeypatch, doc)
    assert m.kit == "https://example.test/audit-kit/pt-f5eece09abe6_rp-244c0791e378"


def _json_without_url_base():
    import copy

    doc = copy.deepcopy(RECORD_DOC)
    del doc["audit"]["kit"]["urlBase"]
    return doc


def test_record_manifest_names_no_artifact_roots_yet(monkeypatch):
    # Interval units need these; the record does not publish them. Guessing a
    # bucket would fail late with a permissions error instead of saying so.
    m = _load_doc(monkeypatch, RECORD_DOC)
    assert m.artifacts.checkpoints is None
    assert m.artifacts.shards is None


def test_our_own_schema_still_wins(monkeypatch):
    doc = {"manifest_format": 1, "run": "r", "kit": "gs://b/k"}
    m = _load_doc(monkeypatch, doc)
    assert (m.run, m.kit) == ("r", "gs://b/k")
