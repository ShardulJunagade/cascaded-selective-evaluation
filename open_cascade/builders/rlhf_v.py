"""openbmb/RLHF-V-Dataset: image + question + human-preferred (chosen) and rejected answers.

Each row becomes one instance with `outputs = [chosen, rejected]` and
`preferences = {"human": 1}`. Images are saved to {out_dir}/images/ so judges can load them
by `image_path`.
"""
import json
from typing import Dict, List, Optional

from open_cascade.builders.base import DatasetBuilder
from open_cascade.config import DataConfig


def _parse_text_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


class RLHFVBuilder(DatasetBuilder):
    """Options:
        dataset_name: HuggingFace dataset id.
        hf_split: which HuggingFace split to read.
        max_samples: optional cap, for quick smoke tests.
    """

    def __init__(self, config: DataConfig, dataset_name: str = "openbmb/RLHF-V-Dataset",
                 hf_split: str = "train", max_samples: Optional[int] = None):
        super().__init__(config)
        self.dataset_name = dataset_name
        self.hf_split = hf_split
        self.max_samples = max_samples

    def load_instances(self) -> List[Dict]:
        from datasets import load_dataset  # HuggingFace datasets; only needed here

        image_dir = self.out_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        dataset = load_dataset(self.dataset_name, split=self.hf_split)
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        instances: List[Dict] = []
        for idx, row in enumerate(dataset):
            text = _parse_text_field(row["text"])
            question = text["question"].strip()
            chosen = text["chosen"].strip()
            rejected = text["rejected"].strip()
            if not question or not chosen or not rejected:
                continue

            image_path = image_dir / f"{idx:06d}.jpg"
            if not image_path.exists():
                row["image"].convert("RGB").save(image_path, quality=95)

            instances.append({
                "image_path": str(image_path),
                "instruction": question,
                "outputs": [chosen, rejected],
                "preferences": {"human": 1},
                "source": {
                    "dataset": self.dataset_name,
                    "row": idx,
                    "origin_dataset": row.get("origin_dataset"),
                    "origin_split": row.get("origin_split"),
                },
            })

        print(f"  loaded {len(instances)} instances from {self.dataset_name}")
        return instances
