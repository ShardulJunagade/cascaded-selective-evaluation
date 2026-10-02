"""Adapter base class.

An adapter's only job is to turn one raw source into canonical `Record`s. It
does not randomize positions, deduplicate, or split - those are shared stages
in `prepare_data/stages.py`, so that every source gets exactly the same
treatment and the funnel table compares like with like.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from prepare_data.config import SOURCES, raw_dir
from prepare_data.funnel import Funnel
from prepare_data.media import MediaStore
from prepare_data.schema import Record

STAGE = "adapted"


class Adapter(ABC):
    """Base for the five source adapters."""

    name: str = ""

    def __init__(self, store: MediaStore, funnel: Funnel,
                 limit: Optional[int] = None, seed: int = 42):
        self.store = store
        self.funnel = funnel
        self.limit = limit
        self.seed = seed
        self.spec: Dict[str, Any] = SOURCES[self.name]
        self.raw = raw_dir(self.name)

    # -- availability -----------------------------------------------------

    def available(self) -> bool:
        """True when the raw download landed. Missing sources are skipped, not fatal."""
        return self.raw.exists() and any(self.raw.rglob("*"))

    def skip_reason(self) -> str:
        return (f"raw data absent at {self.raw}. "
                f"Run `python -m prepare_data.download --source {self.name}`. "
                f"{self.spec.get('note', '')}")

    # -- helpers ----------------------------------------------------------

    def subsample(self, rows: Sequence[Any], limit: Optional[int] = None) -> List[Any]:
        """Reproducible random subset.

        Deliberately *not* `rows[:limit]`: every one of these sources is stored
        in a correlated order (RLHF-V by origin dataset, MM-RLHF by task), so
        taking a prefix would bake a fresh selection bias into the cap.
        """
        limit = self.limit if limit is None else limit
        rows = list(rows)
        if limit is None or limit >= len(rows):
            return rows
        rng = random.Random(f"{self.seed}-{self.name}")
        index = sorted(rng.sample(range(len(rows)), limit))
        return [rows[i] for i in index]

    def hf_dir(self) -> Path:
        return self.raw / "hf"

    def parquet_files(self, pattern: str = "*.parquet") -> List[str]:
        """Parquet shards in the snapshot, ignoring the hub's `.cache` scratch dir."""
        return sorted(
            str(p) for p in self.hf_dir().rglob(pattern)
            if ".cache" not in p.parts
        )

    def load_hf(self, split: Optional[str] = None):
        """Load the snapshotted HF dataset from disk (never re-downloads).

        Loads the parquet shards directly rather than `load_dataset(<dir>)`,
        because a `snapshot_download` directory also holds README.md and the
        hub's `.cache` tree, which the directory loader tries to interpret.
        """
        from datasets import load_dataset

        files = self.parquet_files()
        if not files:
            raise FileNotFoundError(f"No parquet shards under {self.hf_dir()}")
        split = split or self.spec.get("split") or "train"
        return load_dataset("parquet", data_files={split: files}, split=split)

    def keep(self, n: int = 1) -> None:
        self.funnel.keep(STAGE, self.name, n)

    def drop(self, reason: str, n: int = 1) -> None:
        self.funnel.drop(STAGE, self.name, reason, n)

    # -- contract ---------------------------------------------------------

    @abstractmethod
    def iter_records(self) -> Iterator[Record]:
        """Yield canonical records, counting every accepted and rejected row."""
        raise NotImplementedError
