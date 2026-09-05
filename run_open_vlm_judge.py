"""Score one open vision-language judge over a VLM preference split."""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from open_cascade.vlm_registry import VLM_JUDGE_REGISTRY, resolve_vlm_judge


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help=f"VLM judge to run. One of: {sorted(VLM_JUDGE_REGISTRY)}")
    parser.add_argument("--in_filename", type=str, required=True,
                        help="Split to score, e.g. ./data/vlm/split/calibration.jsonl")
    parser.add_argument("--fewshot_in_filename", type=str,
                        default="./data/vlm/split/fewshot.jsonl")
    parser.add_argument("--out_filename", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default="./result/vlm")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Samples per vLLM call; VLM prompts carry images, so keep this modest")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_prefix_caching", action="store_true")
    args = parser.parse_args()

    config = resolve_vlm_judge(args.model_name)
    args.in_filename = Path(args.in_filename)
    if args.out_filename:
        args.out_filename = Path(args.out_filename)
    else:
        args.out_filename = Path(args.result_dir) / f"{args.model_name}.{args.in_filename.stem}.jsonl"

    if args.out_filename.exists() and not args.resume:
        raise SystemExit(f"{args.out_filename} already exists. Pass --resume to continue it.")

    print(f"VLM judge: {args.model_name} -> {config.hf_name}")
    print(f"Input:     {args.in_filename}")
    print(f"Output:    {args.out_filename}")
    return args


if __name__ == "__main__":
    args = parse_args()

    from open_cascade.data import load_fewshot, read_jsonl, sample_key, write_jsonl
    from open_cascade.vlm_judge import OpenVLMJudge

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

    judge = OpenVLMJudge(args.model_name, enable_prefix_caching=not args.no_prefix_caching)
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
