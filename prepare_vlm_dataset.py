"""Prepare a vision-language preference dataset for cascaded selective evaluation.

Default source: openbmb/RLHF-V-Dataset, a compact human-preference dataset where each
sample has an image, a question, a chosen answer, and a rejected answer.
"""
import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Iterable, List, Union


def save_jsonl(samples: Iterable[Dict], out_filename: Union[str, Path]) -> None:
    samples = list(samples)
    if not samples:
        return
    out_filename = Path(out_filename)
    os.makedirs(out_filename.parent, exist_ok=True)
    with open(out_filename, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(json.dumps(sample) for sample in samples) + "\n")


def parse_text_field(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", default="openbmb/RLHF-V-Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", default="./data/vlm")
    parser.add_argument("--N", type=int, default=3, help="N simulated annotators")
    parser.add_argument("--K", type=int, default=3, help="K examples per annotator")
    parser.add_argument("--calibration_set_size", type=int, default=500)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap for quick experiments")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    split_dir = out_dir / "split"
    image_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    samples: List[Dict] = []
    for idx, row in enumerate(dataset):
        text = parse_text_field(row["text"])
        question = text["question"].strip()
        chosen = text["chosen"].strip()
        rejected = text["rejected"].strip()
        if not question or not chosen or not rejected:
            continue

        image_path = image_dir / f"{idx:06d}.jpg"
        if not image_path.exists():
            row["image"].convert("RGB").save(image_path, quality=95)

        samples.append({
            "image_path": str(image_path),
            "instruction": question,
            "outputs": [chosen, rejected],
            "preferences": {"human": 1},
            "source": {
                "dataset": args.dataset_name,
                "row": idx,
                "origin_dataset": row.get("origin_dataset"),
                "origin_split": row.get("origin_split"),
            },
        })

    if len(samples) < args.N * args.K + args.calibration_set_size:
        raise SystemExit(
            f"Need at least N*K + calibration_set_size samples, got {len(samples)}."
        )

    random.seed(args.seed)
    filtered_samples = [s for s in samples if all(len(output) > 0 for output in s["outputs"])]
    fewshot_pool = random.sample(filtered_samples, args.N * args.K)

    fewshot_samples = []
    for annotator_idx in range(args.N):
        start = annotator_idx * args.K
        fewshot_samples.append({
            "annotator": f"human_{annotator_idx}",
            "evaluated_samples": fewshot_pool[start:start + args.K],
        })

    random.shuffle(samples)
    calibration_samples = samples[:args.calibration_set_size]
    test_samples = samples[args.calibration_set_size:]

    save_jsonl(samples, out_dir / "preprocessed.jsonl")
    save_jsonl(fewshot_samples, split_dir / "fewshot.jsonl")
    save_jsonl(calibration_samples, split_dir / "calibration.jsonl")
    save_jsonl(test_samples, split_dir / "test.jsonl")

    print(f"Wrote {len(samples)} samples to {out_dir}")
    print(f"Few-shot:    {split_dir / 'fewshot.jsonl'}")
    print(f"Calibration: {split_dir / 'calibration.jsonl'}")
    print(f"Test:        {split_dir / 'test.jsonl'}")
