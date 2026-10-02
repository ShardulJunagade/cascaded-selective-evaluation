"""jsonl and split helpers shared by the open-model scripts."""
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Union

import jsonlines

# Fields that make up a raw evaluation instance (i.e. everything except judge output).
INSTANCE_FIELDS = ("instruction", "outputs", "preferences")
OPTIONAL_INSTANCE_FIELDS = ("image_path", "source")


def sample_key(sample: Dict) -> str:
    """Identity of an instance.

    Must match the key used by `cascaded_evaluation.util.merge_data`, otherwise judgements
    from different judges will not line up.
    """
    image_path = sample.get("image_path", "")
    return f"{image_path}-{sample['instruction']}-{sample['outputs']}"


def preferred_index(sample: Dict) -> int:
    """Which output the annotators preferred: 1 = outputs[0], 2 = outputs[1].

    Takes the majority over the `preferences` values instead of reading a fixed
    "human" key. The text splits carry a single {"human": n} entry, but the
    multimodal splits key preferences by real annotator id, so that Simulated
    Annotators (Ind.) can group examples by annotator and so that the majority
    is visible rather than pre-collapsed. `cascaded_evaluation.util.prepare_data`
    already derives its labels this way; this keeps few-shot prompt rendering
    consistent with it.
    """
    values = list(sample.get("preferences", {}).values())
    if not values:
        raise ValueError(
            "Few-shot example has no preferences. Tie records are exported with an "
            "empty preferences map and carry no forced choice, so they must not be "
            "used as few-shot demonstrations."
        )
    return Counter(values).most_common(1)[0][0]


def strip_judgement(sample: Dict) -> Dict:
    """Drop `probs`, leaving the raw instance."""
    stripped = {field: sample[field] for field in INSTANCE_FIELDS}
    for field in OPTIONAL_INSTANCE_FIELDS:
        if field in sample:
            stripped[field] = sample[field]
    return stripped


def read_jsonl(path: Union[str, Path]) -> List[Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with jsonlines.open(path) as f:
        return list(f)


def write_jsonl(samples: Iterable[Dict], path: Union[str, Path], mode: str = "w") -> None:
    assert mode in ("w", "a"), "mode should be 'w' or 'a'"
    samples = list(samples)
    if not samples:
        return

    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    # explicit newline so output is byte-identical whether written on Windows or Linux
    with open(path, mode, encoding="utf-8", newline="\n") as f:
        f.write("\n".join(json.dumps(s) for s in samples) + "\n")


def load_fewshot(path: Union[str, Path]) -> List[List[Dict]]:
    """Load one few-shot set per simulated annotator."""
    return [sample["evaluated_samples"] for sample in read_jsonl(path)]


def load_judgements(model_names: List[str], split: str,
                    result_dir: Union[str, Path] = "./result") -> Dict[str, List[Dict]]:
    """Load `{model_name: judged samples}` for one split."""
    result_dir = Path(result_dir)
    return {name: read_jsonl(result_dir / f"{name}.{split}.jsonl") for name in model_names}
