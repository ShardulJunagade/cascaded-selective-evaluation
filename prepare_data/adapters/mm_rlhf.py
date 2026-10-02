"""MM-RLHF: pairs derived from expert rankings.

A raw row is one prompt with several model outputs, each scored on
faithfulness, helpfulness and ethical dimensions, plus a final ranking over the
outputs. There is no pairwise label to read off, so pairs are *constructed*,
and that construction is itself a source of bias: the score-gap threshold tau
decides how easy the resulting comparisons are. Two guards follow from that.

* Only pairs with a score gap >= tau are kept, so near-equal responses are not
  presented as clear preferences. `meta.score_gap` is stored so the difficulty
  distribution can be reported and tau swept.
* Exactly **one pair per prompt** is emitted, chosen at random among the
  qualifying pairs rather than by taking the largest gap. Taking the widest gap
  would systematically select the easiest comparison in every group, and
  emitting all pairs would put correlated items in one split.

Images live in zip archives shipped alongside the metadata; they are extracted
once into `data/raw/mm_rlhf/images/`.
"""

from __future__ import annotations

import random
import zipfile
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

from prepare_data.adapters.base import Adapter
from prepare_data.config import MM_RLHF_TAU
from prepare_data.schema import Record, make_group_id, make_uid

SCORE_FIELDS = ("faithfulness", "helpfulness", "ethical")

# Archive prefixes in the `image` column that the image protocol does not take.
# Safety content is a different judging task with its own ethical dimension,
# and mixing it into a general preference set would confound the agreement
# numbers with refusal behaviour.
EXCLUDED_SUBSETS = {"safety"}


def mean_score(row: Dict, index: int) -> Optional[float]:
    """Overall quality of output `index`: mean of the scored dimensions."""
    values = []
    for field in SCORE_FIELDS:
        series = row.get(field)
        if series is None or index >= len(series) or series[index] is None:
            continue
        values.append(float(series[index]))
    return sum(values) / len(values) if values else None


class MmRlhfAdapter(Adapter):
    name = "mm_rlhf"

    def __init__(self, *args, tau: float = MM_RLHF_TAU, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self._index: Optional[Dict[str, tuple]] = None
        self._handles: List[zipfile.ZipFile] = []

    # -- images -----------------------------------------------------------

    def _archives(self) -> List[Path]:
        """Image archives worth opening: skip video and the excluded subsets."""
        return [
            p for p in sorted(self.hf_dir().rglob("*.zip"))
            if ".cache" not in p.parts
            and p.stem not in EXCLUDED_SUBSETS
            and p.stem != "video"
        ]

    def zip_index(self) -> Dict[str, tuple]:
        """Map each inner path (and basename) to the archive member holding it.

        Images are read straight out of the zips rather than extracted. The
        archives run to tens of GB, and unpacking them would duplicate all of
        that on disk for no benefit: every image is read exactly once and then
        re-encoded into the content-addressed store anyway.
        """
        if self._index is not None:
            return self._index

        index: Dict[str, tuple] = {}
        by_basename: Dict[str, tuple] = {}
        for archive in self._archives():
            try:
                handle = zipfile.ZipFile(archive)
            except (zipfile.BadZipFile, OSError) as exc:
                self.funnel.note(f"mm_rlhf: cannot open {archive.name}: {exc}")
                continue
            self._handles.append(handle)
            for member in handle.namelist():
                if member.endswith("/"):
                    continue
                normalised = member.replace("\\", "/").lstrip("./")
                index.setdefault(normalised, (handle, member))
                # The archives sometimes carry an extra top-level directory
                # relative to the metadata paths, so keep a basename fallback.
                by_basename.setdefault(Path(normalised).name, (handle, member))

        for name, value in by_basename.items():
            index.setdefault(f"basename:{name}", value)

        self._index = index
        self.funnel.note(
            f"mm_rlhf: indexed {len(self._archives())} image archives, "
            f"{sum(1 for k in index if not k.startswith('basename:'))} members, "
            "read in place rather than extracted.")
        return index

    def read_image_bytes(self, relative: str) -> Optional[bytes]:
        relative = (relative or "").strip().lstrip("./").replace("\\", "/")
        if not relative:
            return None
        index = self.zip_index()
        entry = index.get(relative) or index.get(f"basename:{Path(relative).name}")
        if entry is None:
            return None
        handle, member = entry
        try:
            return handle.read(member)
        except (KeyError, zipfile.BadZipFile, OSError) as exc:
            self.funnel.note(f"mm_rlhf: cannot read {member}: {exc}")
            return None

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.close()
            except OSError:
                pass
        self._handles.clear()

    # -- pair construction -------------------------------------------------

    def qualifying_pairs(self, row: Dict) -> List[Dict]:
        outputs: Sequence[str] = row.get("models_output") or []
        if len(outputs) < 2:
            return []

        scores = {i: mean_score(row, i) for i in range(len(outputs))}
        pairs = []
        for i, j in combinations(range(len(outputs)), 2):
            if scores[i] is None or scores[j] is None:
                continue
            gap = abs(scores[i] - scores[j])
            if gap < self.tau:
                continue
            better, worse = (i, j) if scores[i] > scores[j] else (j, i)
            pairs.append({
                "better": better,
                "worse": worse,
                "score_gap": round(gap, 4),
                "score_better": scores[better],
                "score_worse": scores[worse],
            })
        return pairs

    # -- main --------------------------------------------------------------

    def iter_records(self) -> Iterator[Record]:
        dataset = self.load_hf()
        self.funnel.note(
            f"mm_rlhf: {len(dataset)} prompt-level rows; pairs constructed from the "
            f"ranking with tau={self.tau} on the mean of {SCORE_FIELDS}, one pair per prompt."
        )
        self.funnel.note(
            "mm_rlhf: rankings carry no annotator id field, so it supplies neither Ind. "
            "pools nor inter-annotator agreement."
        )
        self.funnel.note(
            "mm_rlhf: the release also ships dpo_pairs.jsonl (62,545 ready-made "
            "chosen/rejected pairs). It is not used here: those pairs are derived from "
            "the same rankings but carry no score gap, so difficulty cannot be "
            "controlled, and several pairs share one prompt, which would put correlated "
            "instances in one split."
        )

        rng = random.Random(f"{self.seed}-mm_rlhf-pairs")
        indices = self.subsample(range(len(dataset)))

        try:
            yield from self._iter_rows(dataset, indices, rng)
        finally:
            self.close()

    def _iter_rows(self, dataset, indices, rng) -> Iterator[Record]:
        for idx in indices:
            row = dataset[int(idx)]

            # Video and safety content are out of scope for the image protocol.
            if row.get("video"):
                self.drop("video prompt")
                continue
            image_field = row.get("image") or ""
            if not image_field:
                self.drop("no image")
                continue
            if image_field.replace("\\", "/").split("/")[0] in EXCLUDED_SUBSETS:
                self.drop("safety subset")
                continue

            question = (row.get("question") or "").strip()
            if not question:
                self.drop("empty question")
                continue

            pairs = self.qualifying_pairs(row)
            if not pairs:
                self.drop(f"no pair with score gap >= {self.tau}")
                continue

            chosen = rng.choice(sorted(pairs, key=lambda p: (p["better"], p["worse"])))
            outputs = row["models_output"]
            better = (outputs[chosen["better"]] or "").strip()
            worse = (outputs[chosen["worse"]] or "").strip()
            if not better or not worse:
                self.drop("empty response in chosen pair")
                continue

            payload = self.read_image_bytes(image_field)
            if payload is None:
                self.drop("image not present in the downloaded archives")
                continue

            ref = self.store.put_bytes(payload, key=f"mm_rlhf:{image_field}")
            if ref is None:
                self.drop("image decode failed")
                continue

            yield Record(
                uid=make_uid(self.name, row.get("id", idx), chosen["better"], chosen["worse"]),
                source=self.name,
                group_id=make_group_id(self.name, [ref.sha256], question),
                modality="image",
                media=[ref.path],
                media_hash=[ref.phash],
                instruction=question,
                response_a=better,
                response_b=worse,
                model_a=f"model_{chosen['better']}",
                model_b=f"model_{chosen['worse']}",
                label="A",
                label_origin="ranking_derived",
                annotations=[],
                n_ann=0,
                agreement=None,
                is_tie=False,
                meta={
                    "row": int(idx),
                    "mm_rlhf_id": row.get("id"),
                    "tau": self.tau,
                    "score_gap": chosen["score_gap"],
                    "score_better": chosen["score_better"],
                    "score_worse": chosen["score_worse"],
                    "n_candidate_pairs": len(pairs),
                    "n_outputs": len(outputs),
                    "len_a": len(better),
                    "len_b": len(worse),
                    "img_res": [ref.width, ref.height],
                    "img_sha256": ref.sha256,
                },
            )
            self.keep()

    # -- for the tau sweep in the bias tables ------------------------------

    def tau_sweep(self, taus: Sequence[float], sample: int = 2000) -> List[Dict]:
        """Pairs surviving each tau, without materialising images."""
        dataset = self.load_hf()
        indices = self.subsample(range(len(dataset)), limit=min(sample, len(dataset)))
        rows = [dataset[int(i)] for i in indices]

        original_tau = self.tau
        results = []
        for tau in taus:
            self.tau = tau
            with_pair = 0
            gaps: List[float] = []
            for row in rows:
                pairs = self.qualifying_pairs(row)
                if pairs:
                    with_pair += 1
                    gaps.append(min(p["score_gap"] for p in pairs))
            results.append({
                "tau": tau,
                "prompts_examined": len(rows),
                "prompts_with_pair": with_pair,
                "fraction": round(with_pair / len(rows), 4) if rows else 0.0,
                "mean_min_gap": round(sum(gaps) / len(gaps), 3) if gaps else None,
            })
        self.tau = original_tau
        return results
