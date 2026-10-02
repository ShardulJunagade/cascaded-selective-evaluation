"""Content-addressed media store.

Images arrive from five sources in five shapes: decoded PIL objects (RLHF-V),
paths into a downloaded archive (MM-RLHF), and plain URLs (VisIT-Bench). They
all end up in one place, keyed by content:

    data/vlm_v2/media/<first two hex chars>/<sha256>.jpg

The sha256 is taken over the *decoded RGB pixels*, not the file bytes, so the
same picture delivered as PNG by one source and JPEG by another lands on one
path and is stored once. That is what makes cross-source dedup possible at all.

Nothing is resized on disk. The judge applies `max_pixels` at prompt time, and
the original resolution is kept in the record so the resolution-sensitivity
ablation has something to slice on.
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from PIL import Image, UnidentifiedImageError

from prepare_data.config import MEDIA_ROOT, OUT_ROOT

# Pillow refuses very large files by default as a decompression-bomb guard.
# Arena screenshots legitimately exceed it, so raise rather than remove it.
Image.MAX_IMAGE_PIXELS = 300_000_000

USER_AGENT = "cascaded-selective-evaluation/1.0"
DEFAULT_FORMAT = "JPEG"
JPEG_QUALITY = 95


@dataclass(frozen=True)
class MediaRef:
    path: str        # relative to data/vlm_v2/, e.g. "media/ab/abcd...jpg"
    sha256: str      # over decoded RGB pixels
    phash: str       # perceptual hash, for near-duplicate detection
    width: int       # original, before any judge-time downscaling
    height: int
    bytes: int

    def to_dict(self) -> Dict:
        return asdict(self)


def pixel_sha256(image: Image.Image) -> str:
    """Hash the decoded pixels, so the digest does not depend on the container."""
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def perceptual_hash(image: Image.Image) -> str:
    import imagehash
    return str(imagehash.phash(image))


class MediaStore:
    """Writes images once, remembers what it has seen, and logs what it lost."""

    def __init__(self, root: Path = MEDIA_ROOT, out_root: Path = OUT_ROOT,
                 image_format: str = DEFAULT_FORMAT, timeout: int = 30,
                 retries: int = 3, index_path: Optional[Path] = None):
        self.root = Path(root)
        self.out_root = Path(out_root)
        self.image_format = image_format.upper()
        self.extension = ".jpg" if self.image_format == "JPEG" else ".png"
        self.timeout = timeout
        self.retries = retries
        self.index_path = Path(index_path) if index_path is not None \
            else self.out_root / "media_index.json"

        self.root.mkdir(parents=True, exist_ok=True)

        # source identity (url / archive path) -> MediaRef, so a URL shared by
        # several rows is fetched once. Persisted between runs, because the
        # content address is only known *after* fetching, so without this every
        # re-run re-downloads every VisIT-Bench URL to rediscover a file it
        # already has.
        self._by_key: Dict[str, MediaRef] = {}
        self._lock = threading.Lock()
        self.n_new = 0
        self.n_reused = 0
        self.n_from_index = 0
        self.failures: List[Dict[str, str]] = []

        self._load_index()

    # -- internals --------------------------------------------------------

    def _load_index(self) -> None:
        """Restore the key -> MediaRef map, dropping entries whose file is gone."""
        if not self.index_path.exists():
            return
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        for key, data in payload.items():
            try:
                ref = MediaRef(**data)
            except TypeError:
                continue          # index written by an older schema
            if (self.out_root / ref.path).exists():
                self._by_key[key] = ref
        self.n_from_index = len(self._by_key)

    def save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {key: ref.to_dict() for key, ref in self._by_key.items()}
        self.index_path.write_text(json.dumps(payload, indent=0), encoding="utf-8")

    def _target(self, sha: str) -> Path:
        return self.root / sha[:2] / f"{sha}{self.extension}"

    def _store(self, image: Image.Image) -> MediaRef:
        rgb = image.convert("RGB")
        sha = pixel_sha256(rgb)
        target = self._target(sha)

        if target.exists():
            with self._lock:
                self.n_reused += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if self.image_format == "JPEG":
                rgb.save(target, "JPEG", quality=JPEG_QUALITY, optimize=True)
            else:
                rgb.save(target, "PNG", optimize=True)
            with self._lock:
                self.n_new += 1

        return MediaRef(
            path=target.relative_to(self.out_root).as_posix(),
            sha256=sha,
            phash=perceptual_hash(rgb),
            width=rgb.width,
            height=rgb.height,
            bytes=target.stat().st_size,
        )

    def _record_failure(self, key: str, reason: str) -> None:
        with self._lock:
            self.failures.append({"key": key[:300], "reason": reason[:300]})

    def _remember(self, key: Optional[str], ref: MediaRef) -> None:
        if key is not None:
            with self._lock:
                self._by_key[key] = ref

    def _cached(self, key: Optional[str]) -> Optional[MediaRef]:
        if key is None:
            return None
        with self._lock:
            ref = self._by_key.get(key)
            if ref is not None:
                self.n_reused += 1
            return ref

    # -- public API -------------------------------------------------------

    def put_image(self, image: Image.Image, key: Optional[str] = None) -> Optional[MediaRef]:
        cached = self._cached(key)
        if cached is not None:
            return cached
        try:
            ref = self._store(image)
        except Exception as exc:                       # noqa: BLE001 - logged, not raised
            self._record_failure(key or "<pil>", f"{type(exc).__name__}: {exc}")
            return None
        self._remember(key, ref)
        return ref

    def put_path(self, path: Union[str, Path]) -> Optional[MediaRef]:
        key = str(path)
        cached = self._cached(key)
        if cached is not None:
            return cached

        source = Path(path)
        if not source.exists():
            self._record_failure(key, "missing file")
            return None
        try:
            with Image.open(source) as image:
                ref = self._store(image)
        except (UnidentifiedImageError, OSError) as exc:
            self._record_failure(key, f"{type(exc).__name__}: {exc}")
            return None

        self._remember(key, ref)
        return ref

    def put_url(self, url: str) -> Optional[MediaRef]:
        cached = self._cached(url)
        if cached is not None:
            return cached

        payload: Optional[bytes] = None
        last_error = "unknown"
        for attempt in range(self.retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                break
            except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries - 1:
                    time.sleep(1.5 * (attempt + 1))

        if payload is None:
            self._record_failure(url, last_error)
            return None

        try:
            with Image.open(io.BytesIO(payload)) as image:
                ref = self._store(image)
        except (UnidentifiedImageError, OSError) as exc:
            self._record_failure(url, f"{type(exc).__name__}: {exc}")
            return None

        self._remember(url, ref)
        return ref

    def prefetch_urls(self, urls: Iterable[str], workers: int = 8) -> int:
        """Fetch many URLs concurrently so the adapter loop does not serialize on IO.

        VisIT-Bench references every image by URL, and fetching them one at a
        time is the slowest part of the whole pipeline. Results land in the
        same cache `put_url` reads, so the adapter code is unchanged.
        """
        pending = []
        with self._lock:
            for url in urls:
                if url and url not in self._by_key and url not in pending:
                    pending.append(url)
        if not pending:
            return 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(self.put_url, pending))
        return len(pending)

    def put_bytes(self, payload: bytes, key: Optional[str] = None) -> Optional[MediaRef]:
        cached = self._cached(key)
        if cached is not None:
            return cached
        try:
            with Image.open(io.BytesIO(payload)) as image:
                ref = self._store(image)
        except (UnidentifiedImageError, OSError) as exc:
            self._record_failure(key or "<bytes>", f"{type(exc).__name__}: {exc}")
            return None
        self._remember(key, ref)
        return ref

    def put(self, obj, key: Optional[str] = None) -> Optional[MediaRef]:
        """Dispatch on whatever shape the adapter happens to hold.

        Covers the five forms these sources deliver: a decoded PIL image, raw
        encoded bytes, the `{"bytes", "path"}` dict a parquet file yields when
        the HF `Image` feature has not been applied, a filesystem path, and a URL.
        """
        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return self.put_image(obj, key=key)
        if isinstance(obj, (bytes, bytearray)):
            return self.put_bytes(bytes(obj), key=key)
        if isinstance(obj, dict):
            if obj.get("bytes"):
                return self.put_bytes(obj["bytes"], key=key)
            if obj.get("path"):
                return self.put_path(obj["path"])
            self._record_failure(key or "<dict>", f"no bytes or path in {sorted(obj)}")
            return None

        text = str(obj)
        if text.startswith(("http://", "https://")):
            return self.put_url(text)
        return self.put_path(text)

    # -- reporting --------------------------------------------------------

    def summary(self) -> Dict:
        files = [f for f in self.root.rglob("*") if f.is_file()]
        return {
            "new": self.n_new,
            "reused": self.n_reused,
            "restored_from_index": self.n_from_index,
            "failed": len(self.failures),
            "unique_on_disk": len(files),
            "bytes_on_disk": sum(f.stat().st_size for f in files),
        }

    def write_failure_log(self, path: Path) -> None:
        if not self.failures:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.failures, indent=2), encoding="utf-8")
