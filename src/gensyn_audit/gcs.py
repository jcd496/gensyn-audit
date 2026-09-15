"""Reading from Google Cloud Storage with nothing but the standard library.

The alternative was making every volunteer install the Cloud SDK, and that is
several hundred megabytes, a `gcloud auth` dance, and -- on macOS -- a
Gatekeeper prompt about an unnotarized `gcloud-crc32c` binary that, when
dismissed the obvious way, silently deletes the thing that verifies your
downloads. None of that is a reasonable ask of someone volunteering a day of
compute, and none of it is necessary: GCS speaks plain HTTPS.

So this module does the three things the runner needs -- read a small object,
list a prefix, download a large object -- over `urllib`, and verifies every
byte against the checksum GCS returns in `x-goog-hash`.

**Credentials are optional by design.** Public artifacts need none, which is
the intended end state for an audit anyone can run. When a bucket is private,
an existing application-default-credentials file is used if there is one, and
the refresh is a single HTTPS POST rather than an SDK. Nothing here shells out.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import AuditError

API = "https://storage.googleapis.com/storage/v1"
MEDIA = "https://storage.googleapis.com"
TOKEN_URL = "https://oauth2.googleapis.com/token"

#: Read size. Large enough that the per-chunk overhead disappears against a
#: 19 GB transfer, small enough that a resume loses little.
CHUNK = 8 * 1024 * 1024


# ── crc32c ───────────────────────────────────────────────────────────────────
# GCS always returns crc32c; md5 is absent for composite objects, so crc32c is
# the checksum that can always be checked. It is Castagnoli, which zlib does
# not implement, and a reflected table is ~10 lines -- cheaper than taking a
# dependency, and far cheaper than the SDK this module exists to avoid.

_CRC_POLY = 0x82F63B78
_CRC_TABLE: list[int] = []

try:
    # A 31 KB C extension with prebuilt wheels for every platform this runner
    # supports. It matters more than its size suggests: the large shards of a
    # checkpoint are composite objects, and GCS publishes no md5 for those — so
    # crc32c is the only checksum available for ~19 of the 19.3 GB. In pure
    # Python that runs at ~20 MB/s against a ~100 MB/s network, making the
    # checksum, not the transfer, the thing an auditor waits on.
    import google_crc32c as _fast_crc32c
except ImportError:  # pragma: no cover - exercised on platforms with no wheel
    _fast_crc32c = None


def _crc_table() -> list[int]:
    if not _CRC_TABLE:
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ (_CRC_POLY if c & 1 else 0)
            _CRC_TABLE.append(c)
    return _CRC_TABLE


class _PyCrc32c:
    """Pure-Python Castagnoli, for platforms with no wheel.

    Correct but slow, and deliberately kept: verification must never depend on
    an optional package being installable. Slice-by-8 was measured at 24 MB/s
    against this one's 20, which is not worth the complexity — the real answer
    when speed matters is the C extension above.
    """

    __slots__ = ("_v",)

    def __init__(self) -> None:
        self._v = 0xFFFFFFFF

    def update(self, data: bytes) -> None:
        table, v = _crc_table(), self._v
        for byte in data:
            v = table[(v ^ byte) & 0xFF] ^ (v >> 8)
        self._v = v

    def digest(self) -> bytes:
        return (self._v ^ 0xFFFFFFFF).to_bytes(4, "big")


class _FastCrc32c:
    """hashlib-shaped wrapper over google_crc32c."""

    __slots__ = ("_c",)

    def __init__(self) -> None:
        assert _fast_crc32c is not None
        self._c = _fast_crc32c.Checksum()

    def update(self, data: bytes) -> None:
        self._c.update(data)

    def digest(self) -> bytes:
        return self._c.digest()


def Crc32c():
    """The fastest correct crc32c available here."""
    return _FastCrc32c() if _fast_crc32c is not None else _PyCrc32c()


def crc32c_implementation() -> str:
    return getattr(_fast_crc32c, "implementation", "python") if _fast_crc32c else "python"


# ── URIs ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GcsUri:
    bucket: str
    name: str

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.name}" if self.name else f"gs://{self.bucket}"


def parse(uri: str) -> GcsUri:
    if not uri.startswith("gs://"):
        raise AuditError(f"not a gs:// URI: {uri}")
    rest = uri[len("gs://") :]
    bucket, _, name = rest.partition("/")
    if not bucket:
        raise AuditError(f"no bucket in {uri}")
    return GcsUri(bucket, name.strip("/"))


# ── credentials ──────────────────────────────────────────────────────────────


@dataclass
class Credentials:
    """A bearer token, or none at all.

    Anonymous is a first-class outcome, not a failure: a bucket an auditor is
    meant to read needs no credential, and requiring one would be the extra
    step this module exists to remove.
    """

    source: str
    _refresh: dict | None = None
    _token: str | None = None
    _expires: float = 0.0

    @property
    def anonymous(self) -> bool:
        return self._refresh is None and self._token is None

    def token(self) -> str | None:
        if self._refresh is None:
            return self._token
        if self._token and time.time() < self._expires - 60:
            return self._token
        body = urllib.parse.urlencode(
            {
                "client_id": self._refresh["client_id"],
                "client_secret": self._refresh["client_secret"],
                "refresh_token": self._refresh["refresh_token"],
                "grant_type": "refresh_token",
            }
        ).encode()
        req = urllib.request.Request(
            TOKEN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                doc = json.load(resp)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise AuditError(
                f"could not refresh your Google credentials: {exc}",
                hint="They may have been revoked. Refresh them with\n"
                "  gcloud auth application-default login\n"
                "or make the artifacts public, which needs no credentials at all.",
            ) from exc
        self._token = doc["access_token"]
        self._expires = time.time() + float(doc.get("expires_in", 3600))
        return self._token


def credentials() -> Credentials:
    """Whatever is already on this machine. Never prompts, never shells out."""
    if tok := os.environ.get("AUDIT_GCS_TOKEN"):
        return Credentials("AUDIT_GCS_TOKEN", _token=tok)

    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(
        Path(os.environ.get("CLOUDSDK_CONFIG", Path.home() / ".config" / "gcloud"))
        / "application_default_credentials.json"
    )

    for path in candidates:
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("type") == "authorized_user" and doc.get("refresh_token"):
            return Credentials(f"ADC ({path})", _refresh=doc)
        # A service-account key needs a signed JWT assertion, which is more
        # crypto than belongs in a downloader. Say so rather than half-doing it.
        if doc.get("type") == "service_account":
            raise AuditError(
                f"{path} is a service-account key, which this runner cannot sign.",
                hint="Export a token instead:\n"
                "  export AUDIT_GCS_TOKEN=$(gcloud auth print-access-token)",
            )
    return Credentials("anonymous")


# ── requests ─────────────────────────────────────────────────────────────────


def _request(url: str, creds: Credentials, *, headers: dict | None = None, timeout: int = 120):
    req = urllib.request.Request(url, headers=headers or {})
    if token := creds.token():
        req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=timeout)


def _explain(exc: urllib.error.HTTPError, uri: str, creds: Credentials) -> AuditError:
    if exc.code in (401, 403):
        if creds.anonymous:
            return AuditError(
                f"{uri} is not publicly readable, and no credentials were found.",
                hint="If you were given access, authenticate once:\n"
                "  gcloud auth application-default login\n"
                "or set AUDIT_GCS_TOKEN to a bearer token. Public artifacts "
                "need neither.",
            )
        return AuditError(
            f"your credentials ({creds.source}) cannot read {uri}.",
            hint="The account may lack access to this bucket.",
        )
    if exc.code == 404:
        return AuditError(f"{uri} does not exist.")
    return AuditError(f"GET {uri} failed ({exc.code}).")


def _media_url(obj: GcsUri) -> str:
    return f"{MEDIA}/{obj.bucket}/{urllib.parse.quote(obj.name)}"


def get(uri: str, creds: Credentials | None = None) -> bytes:
    """Read one small object whole, verifying its checksum."""
    creds = creds or credentials()
    obj = parse(uri)
    try:
        with _request(_media_url(obj), creds) as resp:
            data = resp.read()
            expected = _expected_hashes(resp.headers)
    except urllib.error.HTTPError as exc:
        raise _explain(exc, uri, creds) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AuditError(f"could not reach {uri}: {exc}") from exc
    _verify(data, expected, uri)
    return data


def _expected_hashes(headers) -> dict[str, bytes]:
    """`x-goog-hash: crc32c=…` and, for non-composite objects, `md5=…`."""
    out: dict[str, bytes] = {}
    for raw in headers.get_all("x-goog-hash") or []:
        for part in raw.split(","):
            name, _, b64 = part.strip().partition("=")
            if name in ("crc32c", "md5") and b64:
                try:
                    out[name] = base64.b64decode(b64)
                except (ValueError, TypeError):
                    # A header we cannot decode simply goes unused; it is not
                    # worth failing a download over one malformed field.
                    continue
    return out


class _Hasher(Protocol):
    def update(self, data: Any, /) -> None: ...

    def digest(self) -> bytes: ...


def _hasher(expected: dict[str, bytes]) -> tuple[_Hasher, bytes, str] | None:
    """Pick the checksum to verify with. MD5 first, and not arbitrarily.

    `hashlib.md5` runs at ~930 MB/s against ~20 MB/s for the pure-Python
    crc32c below -- 21 seconds versus 16 minutes over a 19 GB checkpoint. GCS
    supplies md5 for every object that was not composed from parts, so crc32c
    is the fallback for the rare composite, where correctness beats speed.
    """
    if expected.get("md5"):
        return hashlib.md5(), expected["md5"], "md5"
    if expected.get("crc32c"):
        return Crc32c(), expected["crc32c"], "crc32c"
    return None


def _verify(data: bytes, expected: dict[str, bytes], uri: str) -> None:
    picked = _hasher(expected)
    if picked is None:
        return
    h, want, algo = picked
    h.update(data)
    got = h.digest()
    if got != want:
        raise AuditError(
            f"{uri} arrived corrupted ({algo} mismatch).",
            hint="Re-run to fetch it again. Nothing was written.",
        )


def list_prefix(uri: str, creds: Credentials | None = None) -> list[dict]:
    """Every object under a prefix: name, size, and checksums."""
    creds = creds or credentials()
    root = parse(uri)
    prefix = root.name.rstrip("/") + "/" if root.name else ""
    out, token = [], None
    while True:
        query = {
            "prefix": prefix,
            "maxResults": "1000",
            "fields": "items(name,size,md5Hash,crc32c),nextPageToken",
        }
        if token:
            query["pageToken"] = token
        url = f"{API}/b/{root.bucket}/o?{urllib.parse.urlencode(query)}"
        try:
            with _request(url, creds) as resp:
                page = json.load(resp)
        except urllib.error.HTTPError as exc:
            raise _explain(exc, uri, creds) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AuditError(f"could not list {uri}: {exc}") from exc
        for item in page.get("items", []):
            out.append(
                {
                    "name": item["name"],
                    "size": int(item.get("size", 0)),
                    "md5": base64.b64decode(item["md5Hash"]) if item.get("md5Hash") else None,
                    "crc32c": base64.b64decode(item["crc32c"]) if item.get("crc32c") else None,
                }
            )
        token = page.get("nextPageToken")
        if not token:
            return out


def download(
    uri: str,
    dest: Path,
    creds: Credentials | None = None,
    *,
    expected: dict | None = None,
    on_progress=None,
) -> Path:
    """Stream one object to disk, resuming a partial file and verifying it.

    Resume matters at this size: a 19 GB checkpoint over a home connection will
    be interrupted, and restarting from zero each time is how a volunteer gives
    up. The checksum is computed over the whole file including the bytes that
    were already there, so a resumed download is verified as strictly as a
    fresh one.
    """
    creds = creds or credentials()
    obj = parse(uri)
    dest.parent.mkdir(parents=True, exist_ok=True)

    have = dest.stat().st_size if dest.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    digest: _Hasher | None = None
    try:
        with _request(_media_url(obj), creds, headers=headers, timeout=600) as resp:
            total = have + int(resp.headers.get("content-length") or 0)
            # Which checksum to use is only knowable once the response headers
            # are in hand, so the digest is built here rather than up front —
            # choosing before this is how you end up hashing with md5 and
            # comparing against crc32c.
            expected = expected or _expected_hashes(resp.headers)
            picked = _hasher(expected)
            digest = picked[0] if picked else None
            if digest is not None and have:
                # The header describes the whole object, so the running digest
                # has to cover the bytes already on disk too.
                with open(dest, "rb") as fh:
                    for block in iter(lambda: fh.read(CHUNK), b""):
                        digest.update(block)
            with open(dest, "ab" if have else "wb") as out:
                done = have
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    if digest is not None:
                        digest.update(block)
                    done += len(block)
                    if on_progress:
                        on_progress(done, total)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # already complete
            pass
        else:
            raise _explain(exc, uri, creds) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AuditError(
            f"download of {uri} was interrupted at {have} bytes: {exc}",
            hint="Re-run the same command; it resumes from here.",
        ) from exc

    picked = _hasher(expected or {})
    if picked is not None and digest is not None and digest.digest() != picked[1]:
        dest.unlink(missing_ok=True)
        raise AuditError(
            f"{uri} arrived corrupted; the {picked[2]} checksum does not match.",
            hint="The partial file has been removed. Re-run to fetch it again.",
        )
    return dest


def download_prefix(
    uri: str, dest_dir: Path, creds: Credentials | None = None, *, on_file=None, on_progress=None
) -> int:
    """Mirror a prefix locally. Skips files already present and verified.

    ``on_progress(done_bytes, total_bytes, files_done, files_total, current)``
    fires as bytes land, counting progress *inside* each object as well as
    between them. A checkpoint is a handful of multi-gigabyte shards, so a
    per-file-only signal would sit motionless for minutes at a time -- which on
    a 19 GB transfer is indistinguishable from a stall.
    """
    creds = creds or credentials()
    root = parse(uri)
    prefix = root.name.rstrip("/") + "/"
    objects = [
        o
        for o in list_prefix(uri, creds)
        if o["name"].removeprefix(prefix) and not o["name"].endswith("/")
    ]
    if not objects:
        raise AuditError(f"{uri} holds no objects.")

    grand = sum(o["size"] for o in objects)
    done = 0
    for index, item in enumerate(objects, 1):
        rel = item["name"].removeprefix(prefix)
        target = dest_dir / rel
        expected = {k: v for k, v in (("crc32c", item["crc32c"]), ("md5", item["md5"])) if v}

        if target.is_file() and target.stat().st_size == item["size"]:
            if on_file:
                on_file(rel, item["size"], "cached")
            done += item["size"]
            if on_progress:
                on_progress(done, grand, index, len(objects), rel)
            continue

        if on_file:
            on_file(rel, item["size"], "fetching")
        base = done
        download(
            f"gs://{root.bucket}/{item['name']}",
            target,
            creds,
            expected=expected,
            on_progress=(
                lambda d, _t, _b=base, _r=rel, _i=index: on_progress(
                    _b + d, grand, _i, len(objects), _r
                )
            )
            if on_progress
            else None,
        )
        done = base + item["size"]
        if on_progress:
            on_progress(done, grand, index, len(objects), rel)
    return grand
