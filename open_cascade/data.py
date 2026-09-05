"""jsonl and split helpers shared by the open-model scripts."""
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import jsonlines

# Fields that make up a raw evaluation instance (i.e. everything except judge output).
INSTANCE_FIELDS = ("instruction", "outputs", "preferences")


def sample_key(sample: Dict) -> str:
    """Identity of an instance.

    Must match the key used by `cascaded_evaluation.util.merge_data`, otherwise judgements
    from different judges will not line up.
    """
    return f"{sample['instruction']}-{sample['outputs']}"


def strip_judgement(sample: Dict) -> Dict:
    """Drop `probs`, leaving the raw instance."""
    return {field: sample[field] for field in INSTANCE_FIELDS}


def read_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with jsonlines.open(path) as f:
        return list(f)


def write_jsonl(samples: Iterable[Dict], path: str | Path, mode: str = "w") -> None:
    assert mode in ("w", "a"), "mode should be 'w' or 'a'"
    samples = list(samples)
    if not samples:
        return

    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    # explicit newline so output is byte-identical whether written on Windows or Linux
    with open(path, mode, encoding="utf-8", newline="\n") as f:
        f.write("\n".join(json.dumps(s) for s in samples) + "\n")


def load_fewshot(path: str | Path) -> List[List[Dict]]:
    """Load one few-shot set per simulated annotator."""
    return [sample["evaluated_samples"] for sample in read_jsonl(path)]


def load_judgements(model_names: List[str], split: str,
                    result_dir: str | Path = "./result") -> Dict[str, List[Dict]]:
    """Load `{model_name: judged samples}` for one split."""
    result_dir = Path(result_dir)
    return {name: read_jsonl(result_dir / f"{name}.{split}.jsonl") for name in model_names}
