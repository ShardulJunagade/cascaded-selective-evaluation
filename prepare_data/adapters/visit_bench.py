"""VisIT-Bench: instruction-following pairwise judgments.

`visit_bench_human_preferences.csv` holds 5,011 judgment rows over 1,009
(image, instruction, A, B) tuples - about 5 crowdworker judgments each, forced
choice with no tie option. Exactly one of `model_selection.A` /
`model_selection.B` is true on every row.

Two properties shape the adapter:

* **No worker id column.** Judgments on a tuple are indistinguishable rows, so
  we can compute a majority and an agreement level but cannot attribute labels
  to individual annotators. VisIT-Bench therefore supports the abstention-vs-IAA
  analysis but not the Ind. few-shot pools.
* **`human_verified_reference` is one of the seven "models".** Pairs involving
  it are marked `meta.has_reference`, because the strongest model beats the
  reference in only ~27% of comparisons and that skew should be sliceable.

Images are referenced by S3 URL and fetched by the media store.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterator, List, Tuple

from prepare_data.adapters.base import Adapter
from prepare_data.schema import (Annotation, Record, aggregate_annotations,
                                 make_group_id, make_uid)

REFERENCE_MODEL = "human_verified_reference"
TupleKey = Tuple[str, str, str, str, str, str]


def text(value) -> str:
    """Empty string for pandas' NaN, which a few cells in this CSV carry."""
    if value is None or value != value:      # NaN is the only value != itself
        return ""
    return str(value).strip()


class VisitBenchAdapter(Adapter):
    name = "visit_bench"

    def csv_path(self):
        return self.raw / self.spec["filename"]

    def available(self) -> bool:
        return self.csv_path().exists()

    def _group_rows(self) -> Dict[TupleKey, List[bool]]:
        """Collapse judgment rows into {tuple: [prefers_A, ...]}."""
        import pandas as pd

        frame = pd.read_csv(self.csv_path())
        required = {"image_url", "instruction", "A", "B", "A_model", "B_model",
                    "model_selection.A", "model_selection.B"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(
                f"{self.csv_path()} is missing {sorted(missing)}; observed "
                f"{sorted(frame.columns)}. The upstream CSV layout changed."
            )

        # `model_selection.A` is not a valid attribute name for itertuples.
        frame = frame.rename(columns={"model_selection.A": "sel_a",
                                      "model_selection.B": "sel_b"})

        tuples: Dict[TupleKey, List[bool]] = defaultdict(list)
        n_ambiguous = 0
        for row in frame.itertuples(index=False):
            prefers_a = bool(row.sel_a)
            prefers_b = bool(row.sel_b)
            if prefers_a == prefers_b:
                # Neither or both selected: not a usable forced choice.
                n_ambiguous += 1
                continue
            key: TupleKey = (text(row.image_url), text(row.instruction),
                             text(row.A), text(row.B),
                             text(row.A_model), text(row.B_model))
            tuples[key].append(prefers_a)

        if n_ambiguous:
            self.drop("neither/both selected", n_ambiguous)

        self.funnel.note(
            f"visit_bench: {len(frame)} judgment rows collapse to {len(tuples)} tuples "
            "(~5 judgments each). No worker id column, so majority and agreement are "
            "available but Ind. annotator pools are not."
        )
        return tuples

    def iter_records(self) -> Iterator[Record]:
        tuples = self._group_rows()
        keys = self.subsample(sorted(tuples))

        # Every image is a URL, and fetching them inside the loop makes network
        # latency the pipeline's bottleneck. Warm the store concurrently first.
        fetched = self.store.prefetch_urls(key[0] for key in keys)
        self.funnel.note(
            f"visit_bench: {fetched} unique image URLs prefetched for {len(keys)} tuples.")

        for key in keys:
            image_url, instruction, response_a, response_b, model_a, model_b = key
            votes = tuples[key]

            if not image_url or not instruction or not response_a or not response_b:
                self.drop("empty instruction or response")
                continue

            ref = self.store.put_url(image_url)
            if ref is None:
                self.drop("image fetch failed")
                continue

            # Without worker ids the annotators are positional placeholders;
            # they exist so the majority/agreement machinery is shared with the
            # sources that do carry real ids.
            annotations = [
                Annotation(annotator=f"visit_bench_judgment_{i}",
                           label="A" if prefers_a else "B")
                for i, prefers_a in enumerate(votes)
            ]
            summary = aggregate_annotations(annotations)

            yield Record(
                uid=make_uid(self.name, ref.sha256, instruction, response_a, response_b),
                source=self.name,
                group_id=make_group_id(self.name, [ref.sha256], instruction),
                modality="image",
                media=[ref.path],
                media_hash=[ref.phash],
                instruction=instruction,
                response_a=response_a,
                response_b=response_b,
                model_a=str(model_a),
                model_b=str(model_b),
                label=summary["label"],
                label_origin="majority",
                annotations=[a.to_dict() for a in annotations],
                n_ann=summary["n_ann"],
                agreement=summary["agreement"],
                is_tie=summary["is_tie"],
                meta={
                    "image_url": image_url,
                    "has_reference": REFERENCE_MODEL in (model_a, model_b),
                    # VisIT-Bench offers no tie option, so the only way an
                    # instance lands in the tie set is an even split among the
                    # judgments - a different kind of indifference from
                    # VisionArena's explicit "tie" and "tie (bothbad)".
                    "tie_type": "annotator_split" if summary["is_tie"] else None,
                    # The judgment_i names are positional, not identities: the
                    # same index is a different person on every tuple. Ind.
                    # pools must not be built from them.
                    "real_annotator_ids": False,
                    "len_a": len(response_a),
                    "len_b": len(response_b),
                    "img_res": [ref.width, ref.height],
                    "img_sha256": ref.sha256,
                },
            )
            self.keep()
