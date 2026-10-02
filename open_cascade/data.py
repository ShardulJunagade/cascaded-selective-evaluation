"""jsonl, few-shot and split helpers shared by every stage of the pipeline.

An *instance* is one pairwise comparison (see builders/base.py for the full format). A
*judgement* is an instance plus the judge's `probs`: `[P(outputs[0] preferred),
P(outputs[1] preferred)]`, averaged over the simulated annotators.
"""
import json
import os
from collections import Counter
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

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
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def make_splits(samples: List[Dict], n_annotators: int, k_shot: int, calibration_size: int,
                seed: int) -> Tuple[List[List[Dict]], List[Dict], List[Dict]]:
    """Sample N x K few-shot examples, then shuffle the rest into calibration / test.

    Same procedure (and, for a given seed, same result) as the authors'
    `prepare_data_splits.py`. As there, few-shot examples are drawn from the whole pool, so
    they can also appear in calibration or test.

    Returns `(fewshot_sets, calibration, test)`, with one few-shot set per annotator.
    """
    if len(samples) < n_annotators * k_shot + calibration_size:
        raise ValueError(
            f"Need at least N*K + calibration_size = "
            f"{n_annotators * k_shot + calibration_size} samples, got {len(samples)}."
        )

    rng = random.Random(seed)

    # remove trivial samples where either response is empty
    candidates = [s for s in samples if all(len(output) > 0 for output in s["outputs"])]
    pool = rng.sample(candidates, n_annotators * k_shot)
    fewshot_sets = [pool[i * k_shot:(i + 1) * k_shot] for i in range(n_annotators)]

    shuffled = list(samples)
    rng.shuffle(shuffled)
    return fewshot_sets, shuffled[:calibration_size], shuffled[calibration_size:]
