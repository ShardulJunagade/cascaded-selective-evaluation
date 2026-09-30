"""Base class for dataset builders.

A builder turns a raw source (a HuggingFace dataset, released files, ...) into instances in
the common format below, and writes the splits every later stage reads.

Instance format (one JSON object per line):

    instruction   str
    outputs       [response_1, response_2]
    preferences   {annotator_id: 1 | 2}      1 = outputs[0] preferred, 2 = outputs[1]
    image_path    str                        VLM datasets only
    source        dict                       optional provenance

Files written by `build()`:

    {out_dir}/preprocessed/data.jsonl   every instance
    {out_dir}/split/fewshot.jsonl       N sets of K few-shot examples, one per annotator
    {out_dir}/split/calibration.jsonl
    {out_dir}/split/test.jsonl
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from open_cascade.config import DataConfig
from open_cascade.data import make_splits, write_jsonl


class DatasetBuilder(ABC):
    """Subclass this, implement `load_instances()`, and register it in builders/__init__.py.

    Builder-specific settings come from the config's `data.options` and are passed to
    `__init__` as keyword arguments.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.out_dir = Path(config.out_dir)

    @abstractmethod
    def load_instances(self) -> List[Dict]:
        """Return every instance of the dataset in the common format."""

    def build(self) -> None:
        self.check_not_built()
        instances = self.load_instances()

        c = self.config
        fewshot_sets, calibration, test = make_splits(
            instances, c.n_annotators, c.k_shot, c.calibration_size, c.seed)

        # calibration + test is the full dataset, in shuffled order
        write_jsonl(calibration + test, self.out_dir / "preprocessed" / "data.jsonl")
        self.write_fewshot(fewshot_sets)
        self.write_split("calibration", calibration)
        self.write_split("test", test)

    # helpers for subclasses
    def check_not_built(self, splits=("fewshot", "calibration", "test")) -> None:
        """Refuse to overwrite existing splits: judgements scored on them would go stale."""
        existing = [str(self.config.split_file(s)) for s in splits
                    if self.config.split_file(s).exists()]
        if existing:
            raise FileExistsError(
                f"Split files already exist: {existing}. Delete them or choose another "
                f"data.out_dir -- any judgements scored on the old splits would no longer match."
            )

    def write_fewshot(self, fewshot_sets: List[List[Dict]]) -> None:
        records = [
            {"annotator": f"human_{idx}", "evaluated_samples": examples}
            for idx, examples in enumerate(fewshot_sets)
        ]
        write_jsonl(records, self.config.fewshot_file)
        print(f"  wrote {len(records):>5} few-shot sets -> {self.config.fewshot_file}")

    def write_split(self, split: str, samples: List[Dict]) -> None:
        write_jsonl(samples, self.config.split_file(split))
        print(f"  wrote {len(samples):>5} -> {self.config.split_file(split)}")
