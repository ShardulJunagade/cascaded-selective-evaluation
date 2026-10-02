"""Rebuild the text dataset from the judgement files the authors released in ./result/.

The repo ships judgements (`./result/*.jsonl`) but not the underlying data, so there is
nothing to feed a new judge. Each released file carries the full instance next to its
`probs`, so stripping `probs` recovers it.

By default this preserves the *original* calibration/test split, which is what makes a
newly scored open judge directly comparable against the released gpt-4-turbo /
gpt-3.5-turbo results. The few-shot file for that split is the one shipped in
data/split/fewshot.jsonl, so it is not rewritten.

Set `keep_original_split: false` to draw a fresh split instead (e.g. N=K=5); use a new
`data.out_dir` for that, since results are then no longer comparable to the released ones.
"""
from pathlib import Path
from typing import Dict, List, Sequence

from open_cascade.builders.base import DatasetBuilder
from open_cascade.config import DataConfig
from open_cascade.data import read_jsonl, sample_key, strip_judgement, write_jsonl

SPLITS = ("calibration", "test")


class ReleasedTextBuilder(DatasetBuilder):
    """Options:
        reference_model: released judge whose files supply the instances.
        verify_against: other released judges to cross-check instance alignment against.
        result_dir: where the released judgement files are.
        keep_original_split: keep the released calibration/test split (default) or resplit.
    """

    def __init__(self, config: DataConfig, reference_model: str = "gpt-4-turbo",
                 verify_against: Sequence[str] = ("gpt-3.5-turbo", "mistral-7b-instruct"),
                 result_dir: str = "./result", keep_original_split: bool = True):
        super().__init__(config)
        self.reference_model = reference_model
        self.verify_against = list(verify_against)
        self.result_dir = Path(result_dir)
        self.keep_original_split = keep_original_split

    def load_original_splits(self) -> Dict[str, List[Dict]]:
        splits = {
            split: [strip_judgement(s) for s in read_jsonl(
                self.result_dir / f"{self.reference_model}.{split}.jsonl")]
            for split in SPLITS
        }
        keys = {split: {sample_key(s) for s in samples} for split, samples in splits.items()}

        overlap = keys["calibration"] & keys["test"]
        if overlap:
            raise ValueError(f"calibration and test overlap by {len(overlap)} instances")

        for other in self.verify_against:
            for split in SPLITS:
                path = self.result_dir / f"{other}.{split}.jsonl"
                if not path.exists():
                    print(f"  skipping cross-check: {path} not found")
                    continue
                other_keys = {sample_key(s) for s in read_jsonl(path)}
                if other_keys != keys[split]:
                    raise ValueError(
                        f"{other}.{split} covers different instances than "
                        f"{self.reference_model}.{split} ({len(other_keys ^ keys[split])} "
                        f"differ). The splits are not aligned; comparisons would be invalid."
                    )
            print(f"  cross-checked against {other}: identical instances in both splits")

        return splits

    def load_instances(self) -> List[Dict]:
        splits = self.load_original_splits()
        return splits["calibration"] + splits["test"]

    def build(self) -> None:
        if not self.keep_original_split:
            super().build()
            return

        self.check_not_built(splits=SPLITS)
        splits = self.load_original_splits()
        for split in SPLITS:
            self.write_split(split, splits[split])

        pooled = splits["calibration"] + splits["test"]
        write_jsonl(pooled, self.out_dir / "preprocessed" / "data.jsonl")
        print(f"  wrote {len(pooled):>5} -> {self.out_dir / 'preprocessed' / 'data.jsonl'}")
