"""Score one open judge over a split with Simulated Annotators.

Writes `./result/{model_name}.{split}.jsonl`, the same format the authors' scripts produce,
so the output drops straight into their calibration code.

Usage:
    python run_open_judge.py --model_name=qwen2.5-7b-instruct \
        --in_filename=./data/split/calibration.jsonl
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from open_cascade.data import load_fewshot, read_jsonl, sample_key, write_jsonl
from open_cascade.registry import JUDGE_REGISTRY, resolve_judge


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        help=f"Judge to run. One of: {sorted(JUDGE_REGISTRY)}")
    parser.add_argument("--in_filename", type=str, required=True,
                        help="Split to score, e.g. ./data/split/calibration.jsonl")
    parser.add_argument("--fewshot_in_filename", type=str, default="./data/split/fewshot.jsonl",
                        help="Few-shot examples, one set per simulated annotator")

    parser.add_argument("--out_filename", type=str, default=None,
                        help="Override the output path. Use this to re-score a judge whose "
                             "released results you want to keep, e.g. when validating against "
                             "./result/mistral-7b-instruct.*.jsonl")
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Samples per vLLM call; each expands to 2N prompts")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing output file, skipping scored samples")
    parser.add_argument("--no_prefix_caching", action="store_true")

    args = parser.parse_args()

    config = resolve_judge(args.model_name)
    args.in_filename = Path(args.in_filename)

    if args.out_filename:
        args.out_filename = Path(args.out_filename)
    else:
        args.out_filename = Path(f"./result/{args.model_name}.{args.in_filename.stem}.jsonl")

    if args.out_filename.exists() and not args.resume:
        raise SystemExit(f"{args.out_filename} already exists. Pass --resume to continue it.")

    print(f"Judge:  {args.model_name} -> {config.hf_name}")
    print(f"Input:  {args.in_filename}")
    print(f"Output: {args.out_filename}")

    return args


if __name__ == "__main__":
    args = parse_args()

    # imported after arg parsing so --help and validation do not require a GPU
    from open_cascade.judge import OpenJudge

    samples = read_jsonl(args.in_filename)

    if args.resume and args.out_filename.exists():
        done = {sample_key(s) for s in read_jsonl(args.out_filename)}
        before = len(samples)
        samples = [s for s in samples if sample_key(s) not in done]
        print(f"Resuming: {before - len(samples)} already scored, {len(samples)} remaining.")

    if not samples:
        print("Nothing to do.")
        raise SystemExit(0)

    fewshot_examples_list = load_fewshot(args.fewshot_in_filename)
    n_annotators = len(fewshot_examples_list)
    k_shot = len(fewshot_examples_list[0])
    print(f"Simulated Annotators: N={n_annotators}, K={k_shot} "
          f"({2 * n_annotators} forward passes per sample, "
          f"{2 * n_annotators * len(samples)} total)")

    judge = OpenJudge(args.model_name, enable_prefix_caching=not args.no_prefix_caching)
    print(f"Label token ids: {judge.label_token_ids}")

    n_dropped = 0
    try:
        for start in tqdm(range(0, len(samples), args.chunk_size), desc="scoring"):
            chunk = samples[start:start + args.chunk_size]
            probs_list = judge.simulate_annotators_batch(chunk, fewshot_examples_list)

            scored = []
            for sample, probs in zip(chunk, probs_list):
                if probs == [] or np.sum(np.isnan(probs)) > 0:
                    n_dropped += 1
                    continue
                sample["probs"] = probs
                scored.append(sample)

            write_jsonl(scored, args.out_filename, mode="a")
    finally:
        judge.shutdown()

    print(f"Done. Dropped {n_dropped}/{len(samples)} samples with no usable label logprobs.")
