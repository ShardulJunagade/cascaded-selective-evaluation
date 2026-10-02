"""Phase 1: fetch raw data for each source and log what it cost.

Every source writes `data/raw/<source>/download_log.json` recording bytes, wall
time, the resolved revision and any failure. Those numbers are the evidence for
the "what bottleneck did you encounter?" question, so failures are recorded
rather than raised: one gated source must not stop the other four.

Usage:
    python -m prepare_data.download --source all
    python -m prepare_data.download --source rlhf_v visit_bench
    python -m prepare_data.download --source visit_bench --image-limit 1200
"""

from __future__ import annotations

import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional

from prepare_data.config import (ALL_SOURCES, SOURCES, ensure_dirs, raw_dir,
                                 use_utf8_stdout)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def write_log(source: str, log: Dict) -> None:
    out = raw_dir(source) / "download_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2), encoding="utf-8")


def partial_files(source: str) -> List[Dict]:
    """Interrupted transfers still sitting in the hub's download cache.

    Worth reporting: a stale `.incomplete` file means bytes already paid for
    that a plain re-run will resume. Re-running with `--force` instead can make
    the hub open a *new* partial and abandon this one, so the distinction
    matters enough to print.
    """
    cache = raw_dir(source) / "hf" / ".cache" / "huggingface" / "download"
    if not cache.exists():
        return []
    return sorted(
        ({"name": p.name, "bytes": p.stat().st_size, "mtime": p.stat().st_mtime}
         for p in cache.rglob("*.incomplete")),
        key=lambda entry: -entry["bytes"],
    )


def status_of(source: str) -> str:
    """Download state, distinguishing 'never started' from 'still running'.

    The log is only written when a transfer finishes, so its absence alone
    cannot tell the two apart - bytes already on disk can.
    """
    log = read_log(source)
    if log:
        return log.get("status", "unknown")
    directory = raw_dir(source)
    if directory.exists() and any(directory.rglob("*")):
        return "in progress"
    return "not started"


def read_log(source: str) -> Optional[Dict]:
    path = raw_dir(source) / "download_log.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def has_token() -> bool:
    """Whether an HF credential is available, across hub versions.

    `HfFolder.get_token` was removed in huggingface_hub 1.x in favour of the
    module-level `get_token`, which also reads HF_TOKEN from the environment.
    """
    import os

    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token
        return get_token() is not None
    except ImportError:
        pass
    try:
        from huggingface_hub import HfFolder          # hub < 1.0
        return HfFolder.get_token() is not None
    except Exception:                                 # noqa: BLE001
        return False


# --------------------------------------------------------------------------
# downloaders
# --------------------------------------------------------------------------

def download_hf(source: str, spec: Dict) -> Dict:
    """Snapshot an HF dataset repo into data/raw/<source>/hf.

    `ignore_patterns` keeps archives we will never open out of the transfer.
    MM-RLHF, for instance, ships a multi-GB pure-video archive whose rows the
    image protocol discards anyway. Partial files are resumed, so re-running
    after an interrupted transfer continues rather than restarts.
    """
    from huggingface_hub import snapshot_download

    target = raw_dir(source) / "hf"
    target.mkdir(parents=True, exist_ok=True)

    started = time.time()
    path = snapshot_download(
        repo_id=spec["repo"],
        repo_type="dataset",
        local_dir=str(target),
        max_workers=4,
        allow_patterns=spec.get("allow_patterns"),
        ignore_patterns=spec.get("ignore_patterns"),
    )
    elapsed = time.time() - started
    return {
        "status": "ok",
        "path": str(Path(path)),
        "bytes": dir_size(target),
        "seconds": round(elapsed, 1),
        "ignored": spec.get("ignore_patterns"),
    }


def download_url(source: str, spec: Dict) -> Dict:
    """Fetch a single file over HTTP into data/raw/<source>/."""
    import urllib.request

    target = raw_dir(source) / spec["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    request = urllib.request.Request(
        spec["url"], headers={"User-Agent": "cascaded-selective-evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    elapsed = time.time() - started
    return {
        "status": "ok",
        "path": str(target),
        "bytes": len(payload),
        "seconds": round(elapsed, 1),
    }


DOWNLOADERS = {"hf": download_hf, "url": download_url}


def download_source(source: str, force: bool = False) -> Dict:
    spec = SOURCES[source]
    previous = read_log(source)
    if previous and previous.get("status") == "ok" and not force:
        print(f"[{source}] already downloaded ({human(previous['bytes'])}); "
              f"pass --force to refetch")
        return previous

    if spec.get("gated") and not has_token():
        log = {
            "source": source,
            "status": "blocked",
            "reason": "gated",
            "detail": spec["note"],
            "home": spec["home"],
            "bytes": 0,
            "seconds": 0.0,
        }
        print(f"[{source}] BLOCKED (gated). {spec['note']}")
        write_log(source, log)
        return log

    print(f"[{source}] downloading {spec.get('repo') or spec.get('url')} ...")
    try:
        result = DOWNLOADERS[spec["kind"]](source, spec)
    except Exception as exc:                           # noqa: BLE001 - logged, not raised
        log = {
            "source": source,
            "status": "failed",
            "reason": type(exc).__name__,
            "detail": str(exc)[:600],
            "home": spec["home"],
            "bytes": dir_size(raw_dir(source)),
            "seconds": 0.0,
        }
        print(f"[{source}] FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        write_log(source, log)
        return log

    log = {"source": source, "repo": spec.get("repo"), "url": spec.get("url"), **result}
    print(f"[{source}] ok: {human(result['bytes'])} in {result['seconds']}s "
          f"-> {result['path']}")
    write_log(source, log)
    return log


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", default=["all"],
                        help=f"one or more of {ALL_SOURCES}, or 'all'")
    parser.add_argument("--force", action="store_true",
                        help="ignore an existing successful download log and fetch again. "
                             "This is NOT how you resume an interrupted transfer - plain "
                             "re-running resumes, whereas --force can make the hub start "
                             "a new partial file and abandon the one on disk.")
    return parser.parse_args()


def main(sources: List[str], force: bool = False) -> Dict[str, Dict]:
    use_utf8_stdout()
    ensure_dirs()
    if "all" in sources:
        sources = ALL_SOURCES

    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise SystemExit(f"Unknown source(s) {unknown}. Known: {ALL_SOURCES}")

    logs = {source: download_source(source, force=force) for source in sources}

    print("\n" + "=" * 62)
    print(f"{'source':<16}{'status':<10}{'size':>12}{'seconds':>10}")
    print("-" * 62)
    for source, log in logs.items():
        print(f"{source:<16}{log['status']:<10}{human(log.get('bytes', 0)):>12}"
              f"{log.get('seconds', 0):>10.1f}")
    print("=" * 62)

    blocked = [s for s, log in logs.items() if log["status"] != "ok"]
    if blocked:
        print("\nNot downloaded:")
        for source in blocked:
            print(f"  - {source}: {logs[source].get('reason')} "
                  f"({logs[source].get('home')})")
        print("These are recorded in data/raw/<source>/download_log.json and reported "
              "as bottlenecks; the pipeline runs on whatever did arrive.")
    return logs


if __name__ == "__main__":
    args = parse_args()
    main(args.source, force=args.force)
