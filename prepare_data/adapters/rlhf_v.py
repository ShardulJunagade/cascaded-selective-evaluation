"""RLHF-V: fine-grained correctional human feedback.

Each raw row is an image, a question, a model response, and a human-edited
version of that response with the hallucinated spans corrected. The edit is by
construction the preferred side, which makes this the most skewed of the five
sources - hence `label_origin="correction"` and pilot status rather than a
headline result.

It also carries no annotator ids and one annotation per instance, so it cannot
feed the Simulated Annotators (Ind.) pools or the abstention-vs-IAA analysis.
Both facts are recorded as funnel notes.
"""

from __future__ import annotations

import json
from typing import Iterator

from prepare_data.adapters.base import Adapter
from prepare_data.schema import Record, make_group_id, make_uid

HUMAN = "human_corrected"


def parse_text(value) -> dict:
    return json.loads(value) if isinstance(value, str) else value


class RlhfVAdapter(Adapter):
    name = "rlhf_v"

    def iter_records(self) -> Iterator[Record]:
        dataset = self.load_hf()
        self.funnel.note(
            f"rlhf_v: HF release `{self.spec['repo']}` has {len(dataset)} rows. The 1.4K "
            "figure quoted in the Assignment 1 report is the original paper's annotation "
            "count; the published release is larger and is what both the old "
            "prepare_vlm_dataset.py (5,732 samples) and this pipeline read."
        )
        self.funnel.note(
            "rlhf_v: no annotator ids and one annotation per instance, so it cannot "
            "supply Ind. few-shot pools or inter-annotator agreement."
        )

        indices = self.subsample(range(len(dataset)))
        for idx in indices:
            row = dataset[int(idx)]

            try:
                text = parse_text(row["text"])
                question = (text.get("question") or "").strip()
                chosen = (text.get("chosen") or "").strip()
                rejected = (text.get("rejected") or "").strip()
            except (json.JSONDecodeError, AttributeError, TypeError):
                self.drop("unparsable text field")
                continue

            if not question or not chosen or not rejected:
                self.drop("empty question or response")
                continue

            ref = self.store.put(row["image"], key=f"rlhf_v:{idx}")
            if ref is None:
                self.drop("image decode failed")
                continue

            origin = row.get("origin_dataset") or row.get("ds_name") or "unknown"

            # Canonical pre-randomization order: preferred side first. The
            # position-randomization stage in stages.py is what actually breaks
            # the ordering, for every source alike.
            yield Record(
                uid=make_uid(self.name, idx, ref.sha256),
                source=self.name,
                group_id=make_group_id(self.name, [ref.sha256], question),
                modality="image",
                media=[ref.path],
                media_hash=[ref.phash],
                instruction=question,
                response_a=chosen,
                response_b=rejected,
                model_a=HUMAN,
                model_b=str(origin),
                label="A",
                label_origin="correction",
                annotations=[],
                n_ann=0,
                agreement=None,
                is_tie=False,
                meta={
                    "row": int(idx),
                    "ds_name": row.get("ds_name"),
                    "origin_dataset": row.get("origin_dataset"),
                    "origin_split": row.get("origin_split"),
                    "len_a": len(chosen),
                    "len_b": len(rejected),
                    "img_res": [ref.width, ref.height],
                    "img_sha256": ref.sha256,
                },
            )
            self.keep()
