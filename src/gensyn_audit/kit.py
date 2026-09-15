"""The audit kit: pinned wheels, their digests, and the trajectory they verify.

An audit is a claim about a specific pair of commits -- the pretrain code and
the repop kernels -- as much as about a hash. A kit binds them: one immutable
prefix holding ``kit.json`` (the two commits plus a sha256 for every file), the
wheels themselves, and ``trajectory.json`` (the canonical hashes those wheels
were minted against). Published by ``scripts/audit_kit/publish_audit_kit.sh``
to ``gs://gensyn-audit-artifacts/audit-kit/<kit-id>/`` and read from there
over plain HTTPS -- no Cloud SDK on the auditor's machine.

This module owns the provenance chain, which is five links and only as strong
as its weakest:

1. every downloaded file's sha256 matches ``kit.json``  -- before install
2. ``repop.build_info()["commit"]`` equals ``kit.json.repop_commit``
3. the build carries a backend that can serve the device (upstream's
   ``_require_repop_backend`` enforces this too, and also rejects an *unstamped*
   build whose commit reads "unknown")
4. ``trajectory.json.repop_commit`` equals ``kit.json.repop_commit`` -- hashes
   and wheels must pair, or the digests describe kernels you did not run
5. the replay result's ``repop.commit`` equals the kit's -- checked after the
   fact, because a result naming a different build verified different kernels
   and proves nothing about the published trajectory

Nothing here builds from source. That path exists (docs/mps-audit-runbook.md)
and needs Homebrew libomp and a baked rpath; the kit exists so an auditor never
has to touch it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .errors import AuditError

SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: repop wheels are built per CPython minor; the published ones are cp311.
REQUIRED_PYTHON = (3, 11)


# ── kit.json ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KitFile:
    name: str
    sha256: str
    bytes: int

    @property
    def is_wheel(self) -> bool:
        return self.name.endswith(".whl")

    @property
    def is_repop(self) -> bool:
        return self.is_wheel and self.name.startswith("repop-")


@dataclass(frozen=True)
class Kit:
    """A resolved ``kit.json``."""

    kit_format: int
    pretrain_commit: str
    repop_commit: str
    files: tuple[KitFile, ...]
    prefix: str
    """Where it came from, so the receipt can name a resolvable kit URL."""

    public_url_base: str | None = None
    """The https prefix the publisher served this kit from.

    A kit read over `gs://` is reachable only with Google credentials, and a
    volunteer has none. The publisher writes the same bytes to a public bucket
    and records that URL here, so a `gs://` reference can still yield something
    an auditor can fetch -- and a receipt can cite a URL anyone can open.
    """

    @property
    def public_prefix(self) -> str:
        """Where to read this kit from. The https mirror when there is one."""
        return self.public_url_base or self.prefix

    @property
    def kit_id(self) -> str:
        return f"pt-{self.pretrain_commit[:12]}_rp-{self.repop_commit[:12]}"

    def file(self, name: str) -> KitFile:
        for f in self.files:
            if f.name == name:
                return f
        raise AuditError(f"{name} is not listed in this kit's manifest.")

    def wheels_in_install_order(self, platform_tag: str) -> tuple[KitFile, ...]:
        """repop first: its wheel pins the torch the hashes were produced under.

        Installing pretrain first lets pip resolve a different torch, and the
        published digests were minted against exactly one.
        """

        def compatible(file: KitFile) -> bool:
            python_tag, abi_tag, wheel_platform = file.name[:-4].rsplit("-", 3)[-3:]
            if not {"py3", "py311", "cp311"}.intersection(python_tag.split(".")):
                return False
            if abi_tag not in {"none", "cp311"}:
                return False
            if wheel_platform == "any":
                return True
            return any(
                (platform_tag == "macosx" and tag.startswith("macosx_") and tag.endswith("_arm64"))
                or (platform_tag == "linux" and "linux" in tag and tag.endswith("_x86_64"))
                for tag in wheel_platform.split(".")
            )

        repop = [f for f in self.files if f.is_repop and compatible(f)]
        if not repop:
            available = sorted(f.name for f in self.files if f.is_repop)
            raise AuditError(
                f"this kit has no repop wheel for {platform_tag}.",
                hint="Available: "
                + (", ".join(available) or "none")
                + "\nThe macOS audit wheel is repop-*-macosx_*_arm64.whl.",
            )
        groups: dict[str, list[KitFile]] = {}
        for file in self.files:
            if file.is_wheel:
                groups.setdefault(file.name.split("-", 1)[0], []).append(file)
        selected = {}
        for distribution, candidates in groups.items():
            matches = [file for file in candidates if compatible(file)]
            if len(matches) != 1:
                raise AuditError(
                    f"expected one {distribution} wheel for {platform_tag}, found {len(matches)}.",
                    hint="The kit must contain exactly one compatible wheel per package.",
                )
            selected[distribution] = matches[0]
        return (selected.pop("repop"), *selected.values())


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ── trajectory.json ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Unit:
    """One independently verifiable claim in the published trajectory."""

    kind: str
    """``init`` (no checkpoint, no data) or ``interval`` (one training step)."""

    config_name: str
    until_step: int
    state_hash: str
    devices_verified: tuple[str, ...]
    mps_wall_seconds: float | None
    mps_peak_rss_gb: float | None
    checkpoint_uri: str | None
    gcs_root: str | None
    expect_hash: str | None
    descriptor_uri: str | None = None
    """Set only for the one interval that crosses a fork boundary."""

    predecessor_step: int | None = None
    """The step the predecessor checkpoint IS — not `until_step - 1`.

    Checkpoints are saved every `ckpt_every_tokens`, not every step, so a
    replay to step 200 commonly starts from step 100. Deriving it would name a
    directory that was never written.
    """

    @property
    def is_init(self) -> bool:
        return self.kind == "init"

    @property
    def label(self) -> str:
        return f"{self.config_name} {self.kind}" + (
            "" if self.is_init else f" → step {self.until_step}"
        )

    @property
    def target_hash(self) -> str:
        """What the replay must reproduce. For an init unit the state hash IS
        the target; an interval unit names the next step's digest separately."""
        return self.expect_hash or self.state_hash


@dataclass(frozen=True)
class Trajectory:
    trajectory_format: int
    name: str
    repop_commit: str
    pretrain_commit: str
    notes: str
    units: tuple[Unit, ...]

    @property
    def init_units(self) -> tuple[Unit, ...]:
        return tuple(u for u in self.units if u.is_init)

    def find(self, config_name: str, kind: str | None = None) -> Unit:
        for u in self.units:
            if u.config_name == config_name and (kind is None or u.kind == kind):
                return u
        names = ", ".join(sorted({u.config_name for u in self.units}))
        raise AuditError(
            f"no {kind or 'unit'} for {config_name!r} in trajectory {self.name!r}.",
            hint=f"This trajectory covers: {names}",
        )


# ── fetching ─────────────────────────────────────────────────────────────────


def _raise_not_a_document(uri: str, final_uri: str) -> None:
    """`uri` answered with a web page. Say which wall was hit, and how to pass it."""
    redirected = final_uri and final_uri.split("?", 1)[0] != uri
    where = f"\n  It redirected to {final_uri.split('?', 1)[0]}" if redirected else ""
    raise AuditError(
        f"{uri} returned a web page, not a document.{where}",
        hint="This is almost always an access gateway asking you to sign in.\n"
        "Staging sits behind Cloudflare Access; production is not expected to.\n"
        "Get a token and put it in the environment:\n"
        '  export AUDIT_HTTP_HEADERS="cf-access-token: $(cloudflared access token \\\n'
        '      -app=https://open1b.gensyn-staging.ai)"\n'
        "Or point --manifest at a record that does not need one.",
    )


def _read(uri: str) -> bytes:
    """Read one small file from gs://, https:// or a local path."""
    if uri.startswith("gs://"):
        from . import gcs

        return gcs.get(uri)

    if uri.startswith(("http://", "https://")):
        # Same `AUDIT_HTTP_HEADERS` seam the record uses: a manifest served from
        # behind Cloudflare Access is unreadable without it, and the web app
        # prints `--manifest <record>/manifest.json`.
        from . import __version__
        from .record import _extra_headers

        req = urllib.request.Request(uri, headers=_extra_headers())
        # Cloudflare rejects the default `Python-urllib/…` agent outright.
        req.add_header("User-Agent", f"gensyn-audit/{__version__}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                # An access gateway answers with its login page, not an error
                # status, so the failure would otherwise surface as "not valid
                # JSON" -- which sends people looking at the file instead of at
                # their credentials. Nothing this reads is ever HTML: not a
                # manifest, not kit.json, not a wheel.
                if resp.headers.get_content_type() == "text/html":
                    _raise_not_a_document(uri, resp.geturl())
                return body
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AuditError(f"could not fetch {uri}: {exc}") from exc
    path = Path(uri.removeprefix("file://")).expanduser()
    if not path.is_file():
        raise AuditError(f"no such file: {path}")
    return path.read_bytes()


def local_uri_path(uri: str) -> Path | None:
    """The local path a URI names, or None if it is remote.

    One function, so the planner and the fetcher can never disagree about where
    a checkpoint is. They disagreeing points --checkpoint at a directory nothing
    populates, and the replay dies hours later on a missing file.
    """
    if uri.startswith("gs://"):
        return None
    return Path(uri.removeprefix("file://")).expanduser()


def _join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def load_kit(prefix: str) -> Kit:
    """Read and validate ``<prefix>/kit.json``."""
    raw = _read(_join(prefix, "kit.json"))
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{prefix}/kit.json is not valid JSON: {exc}") from exc

    fmt = doc.get("kit_format")
    if fmt != 1:
        raise AuditError(
            f"kit at {prefix} declares kit_format {fmt}; this runner understands 1.",
            hint="pip install -U gensyn-audit",
        )
    for key in ("pretrain_commit", "repop_commit"):
        value = str(doc.get(key, ""))
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise AuditError(f"kit at {prefix}: {key} is not a full commit sha ({value!r}).")

    files = []
    for entry in doc.get("files", []):
        name, sha = str(entry.get("name", "")), str(entry.get("sha256", "")).lower()
        if not name or not SHA256.match(sha):
            raise AuditError(f"kit at {prefix}: file entry {entry!r} is missing a name or sha256.")
        files.append(KitFile(name=name, sha256=sha, bytes=int(entry.get("bytes", 0))))
    if not files:
        raise AuditError(f"kit at {prefix} lists no files.")

    public = str(doc.get("public_url_base", "")).rstrip("/") or None
    if public and not public.startswith(("http://", "https://")):
        # A gs:// value here would defeat the point: the field exists to name a
        # location that needs no credentials.
        raise AuditError(f"kit at {prefix}: public_url_base is not an http(s) URL ({public!r}).")

    return Kit(
        kit_format=fmt,
        pretrain_commit=doc["pretrain_commit"],
        repop_commit=doc["repop_commit"],
        files=tuple(files),
        prefix=prefix.rstrip("/"),
        public_url_base=public,
    )


def load_trajectory(kit: Kit, *, name: str = "trajectory.json") -> Trajectory:
    """Read ``<prefix>/trajectory.json`` and enforce provenance link 4."""
    raw = _read(_join(kit.public_prefix, name))
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AuditError(f"{kit.prefix}/{name} is not valid JSON: {exc}") from exc

    if doc.get("trajectory_format") != 1:
        raise AuditError(
            f"trajectory declares format {doc.get('trajectory_format')}; this runner understands 1."
        )

    # Link 4. The publisher refuses to bundle a mismatched pair, but a kit can
    # be assembled by other means and the consequence is silent and total: the
    # digests would describe kernels the wheels do not contain.
    if doc.get("repop_commit") != kit.repop_commit:
        raise AuditError(
            "this trajectory was minted against a different repop build than the "
            "kit bundles.\n"
            f"    trajectory  {doc.get('repop_commit')}\n"
            f"    kit         {kit.repop_commit}",
            hint="Hashes and wheels must pair. Verifying against these digests would "
            "compare your kernels to someone else's.",
        )

    units = []
    for raw_unit in doc.get("units", []):
        units.append(
            Unit(
                kind=str(raw_unit.get("kind", "init")),
                config_name=str(raw_unit["config_name"]),
                until_step=int(raw_unit.get("until_step", 0)),
                state_hash=str(raw_unit["state_hash"]).lower(),
                devices_verified=tuple(raw_unit.get("devices_verified", ())),
                mps_wall_seconds=raw_unit.get("mps_wall_seconds"),
                mps_peak_rss_gb=raw_unit.get("mps_peak_rss_gb"),
                checkpoint_uri=raw_unit.get("checkpoint_uri"),
                gcs_root=raw_unit.get("gcs_root"),
                expect_hash=(
                    str(raw_unit["expect_hash"]).lower() if raw_unit.get("expect_hash") else None
                ),
            )
        )
    if not units:
        raise AuditError(f"trajectory {doc.get('name')!r} lists no units.")

    return Trajectory(
        trajectory_format=1,
        name=str(doc.get("name", "unnamed")),
        repop_commit=kit.repop_commit,
        pretrain_commit=str(doc.get("pretrain_commit", kit.pretrain_commit)),
        notes=str(doc.get("notes", "")),
        units=tuple(units),
    )


# ── staging: provenance link 1 ───────────────────────────────────────────────


def stage(
    kit: Kit, dest: Path, *, only: tuple[KitFile, ...] | None = None, on_progress=None
) -> dict[str, Path]:
    """Download the kit's files and verify every sha256 before anything runs.

    Idempotent: a file already present with the right digest is not re-fetched,
    so an interrupted install resumes for free. A file present with the WRONG
    digest is deleted and re-fetched rather than trusted -- it is far more
    likely to be a truncated download than an attack, and either way it must
    not be installed.
    """
    dest.mkdir(parents=True, exist_ok=True)
    wanted = only if only is not None else kit.files
    staged: dict[str, Path] = {}

    for entry in wanted:
        path = dest / entry.name
        if path.is_file() and _digest(path) == entry.sha256:
            if on_progress:
                on_progress(entry, "cached")
            staged[entry.name] = path
            continue
        if path.exists():
            path.unlink()
        if on_progress:
            on_progress(entry, "fetching")
        path.write_bytes(_read(_join(kit.public_prefix, entry.name)))

        got = _digest(path)
        if got != entry.sha256:
            path.unlink(missing_ok=True)
            raise AuditError(
                f"{entry.name} does not match the digest the kit publishes.\n"
                f"    downloaded  {got}\n"
                f"    kit.json    {entry.sha256}",
                hint="Refusing to install it. Re-run to retry the download; if it "
                "keeps failing, the kit and the bucket disagree, which is worth "
                "reporting rather than working around.",
            )
        if on_progress:
            on_progress(entry, "verified")
        staged[entry.name] = path

    return staged


# ── provisioning ─────────────────────────────────────────────────────────────


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def replay_entrypoint(venv_dir: Path) -> Path:
    """The console script the wheel installs. Not ``python -m`` over a checkout."""
    return venv_dir / "bin" / "pretrain-audit-replay"


def verify_entrypoint(venv_dir: Path) -> Path:
    """The hand-off verifier, from the same wheel as the replay.

    Reconstructing the v3 state hash means loading the model and the optimizer,
    so it belongs to the pinned pretrain the digests were minted against —
    a second implementation in this CLI would be a second definition of the
    hash the whole record rests on.
    """
    return venv_dir / "bin" / "pretrain-audit-verify-handoff"


def find_base_interpreter() -> str | None:
    """A CPython matching the wheels' ABI tag, or None if there is none.

    The published repop wheels are cp311, so 3.11 is not a preference. Prefer
    an explicit python3.11 on PATH, then our own interpreter when it is 3.11.

    With ``uv tool install --python 3.11``, the tool runs on 3.11 inside its
    own environment even when nothing named ``python3.11`` is on PATH.
    """
    if found := shutil.which("python3.11"):
        return found
    if sys.version_info[:2] == REQUIRED_PYTHON:
        return sys.executable
    return None


def _base_interpreter() -> str:
    """The interpreter to build the kit venv with, or a refusal that says why."""
    if interpreter := find_base_interpreter():
        return interpreter
    raise AuditError(
        f"no python{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} on PATH, and this runner "
        f"is running under {sys.version_info.major}.{sys.version_info.minor}.",
        hint="The published repop wheels are built for CPython 3.11 (cp311 ABI tag) "
        "and will not install on another minor version. Install the tool on "
        "3.11 and uv brings its own:\n"
        "  uv tool install --python 3.11 gensyn-audit",
    )


def provision(
    kit: Kit, venv_dir: Path, staged: dict[str, Path], *, platform_tag: str, on_step=None
) -> Path:
    """Create the audit venv and install the kit's wheels into it.

    Returns the venv's interpreter. Safe to re-run: an existing venv whose
    installed repop already reports the kit's commit is left alone, so a
    resumed ``gensyn-audit run`` costs one import instead of a reinstall.
    """
    wheels = kit.wheels_in_install_order(platform_tag)
    python = venv_python(venv_dir)

    if python.is_file():
        try:
            info = build_info(python)
        except AuditError:
            info = None
        if (
            info
            and info.get("commit") == kit.repop_commit
            and replay_entrypoint(venv_dir).is_file()
        ):
            if on_step:
                on_step("venv already provisioned for this kit")
            return python

    if on_step:
        on_step(f"creating venv at {venv_dir}")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [_base_interpreter(), "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AuditError(
            f"could not create a venv at {venv_dir}",
            hint=(proc.stderr or proc.stdout).strip()[-500:],
        )

    for wheel in wheels:
        if on_step:
            on_step(f"installing {wheel.name}")
        proc = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                str(staged[wheel.name]),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AuditError(
                f"could not install {wheel.name}",
                hint=(proc.stderr or proc.stdout).strip()[-900:],
            )
    return python


def build_info(python: Path) -> dict:
    """``repop.build_info()`` from inside the venv: commit, backends, arch list."""
    proc = subprocess.run(
        [str(python), "-c", "import repop, json; print(json.dumps(repop.build_info()))"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AuditError(
            "could not read repop.build_info() from the audit venv.",
            hint=(proc.stderr or proc.stdout).strip()[-500:],
        )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise AuditError(f"repop.build_info() returned no JSON: {proc.stdout[:300]!r}") from exc


def verify_build(python: Path, kit: Kit, *, device: str) -> dict:
    """Provenance links 2 and 3, checked before a replay is allowed to start.

    Upstream's ``_require_repop_backend`` also enforces link 3, and would raise
    on its own. Checking here means the auditor learns it in the preflight
    rather than after a download, and learns *which* kit the build disagrees
    with -- which upstream cannot know.
    """
    info = build_info(python)

    if info.get("commit") != kit.repop_commit:
        raise AuditError(
            "the installed repop is not the build this kit pins.\n"
            f"    installed  {info.get('commit')}\n"
            f"    kit.json   {kit.repop_commit}",
            hint="A result naming a different kernel build verified different "
            "kernels and proves nothing about the published trajectory. "
            "Delete the audit venv and re-run to reinstall.",
        )

    needed = {"mps": "metal", "cuda": "cuda", "cpu": "cpu"}[device]
    backends = info.get("backends") or []
    if needed not in backends:
        raise AuditError(
            f"--device {device} needs repop's {needed!r} backend, and this build "
            f"has {backends or 'none'}.",
            hint="A build without it falls back to CPU kernels op by op and would "
            "'pass' without exercising the hardware the verification claims to "
            "cover. Install the platform's audit wheel, or verify on --device cpu "
            "explicitly.",
        )
    return info


def verify_result_provenance(result: dict, kit: Kit) -> None:
    """Provenance link 5: the result must name the build the kit pins."""
    got = (result.get("repop") or {}).get("commit")
    if got != kit.repop_commit:
        raise AuditError(
            "the replay result names a different repop build than the kit.\n"
            f"    result    {got}\n"
            f"    kit.json  {kit.repop_commit}",
            hint="This result verified different kernels. It is not evidence about "
            "the published trajectory and must not be submitted as such.",
        )
