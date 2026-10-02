"""`score` stage: run one judge over the dataset splits and cache its judgements.

Writes `{result_dir}/{judge}.{split}.jsonl`, the same format the authors' scripts produce, so
the output also drops straight into their calibration code. Scoring is resumable: already
scored instances are skipped, so a preempted job continues where it stopped.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from open_cascade.config import ExperimentConfig
from open_cascade.data import load_fewshot, read_jsonl, sample_key, write_jsonl
from open_cascade.registry import resolve_judge


def pending_samples(in_file: Path, out_file: Path) -> List[Dict]:
    """Instances in `in_file` that are not yet in `out_file`."""
    samples = read_jsonl(in_file)
    if not out_file.exists():
        return samples

    done = {sample_key(s) for s in read_jsonl(out_file)}
    remaining = [s for s in samples if sample_key(s) not in done]
    print(f"  resuming: {len(samples) - len(remaining)} already scored, "
          f"{len(remaining)} remaining")
    return remaining


def is_usable(probs: List[float]) -> bool:
    return not (probs == [] or np.sum(np.isnan(probs)) > 0)


def score_judge(config: ExperimentConfig, judge_name: str) -> None:
    """Score `judge_name` on every split in `scoring.splits`. Loads the model at most once."""
    from open_cascade.judges import build_judge  # needs vLLM; imported only when scoring

    scoring = config.scoring
    judge_config = resolve_judge(judge_name)

    # check every output file before spending minutes loading the model
    if not scoring.resume:
        for split in scoring.splits:
            out_file = config.result_file(judge_name, split)
            if out_file.exists():
                raise SystemExit(f"{out_file} already exists. Set scoring.resume=true to "
                                 f"continue it, or delete it to start over.")

    fewshot_examples_list = load_fewshot(config.data.fewshot_file)
    n_annotators = len(fewshot_examples_list)
    k_shot = len(fewshot_examples_list[0])
    if scoring.max_fewshot_examples is not None:
        k_shot = min(k_shot, scoring.max_fewshot_examples)

    print(f"Judge:   {judge_name} -> {judge_config.hf_name}")
    print(f"Simulated Annotators: N={n_annotators}, K={k_shot} "
          f"({2 * n_annotators} forward passes per sample)")

    judge_kwargs = dict(enable_prefix_caching=scoring.prefix_caching,
                        max_fewshot_examples=k_shot)
    if scoring.released_mistral_compat:
        judge_kwargs["released_mistral_compat"] = True

    judge = None
    try:
        for split in scoring.splits:
            in_file = config.data.split_file(split)
            out_file = config.result_file(judge_name, split)
            print(f"\n[{split}] {in_file} -> {out_file}")

            samples = pending_samples(in_file, out_file)
            if not samples:
                print("  nothing to do")
                continue

            if judge is None:
                judge = build_judge(judge_name, **judge_kwargs)
                print(f"  label token ids: {judge.label_token_ids}")

            n_dropped = 0
            for start in tqdm(range(0, len(samples), scoring.chunk_size), desc=split):
                chunk = samples[start:start + scoring.chunk_size]
                probs_list = judge.simulate_annotators_batch(chunk, fewshot_examples_list)

                scored = []
                for sample, probs in zip(chunk, probs_list):
                    if not is_usable(probs):
                        n_dropped += 1
                        continue
                    sample["probs"] = probs
                    scored.append(sample)

                write_jsonl(scored, out_file, mode="a")

            print(f"  done; dropped {n_dropped}/{len(samples)} samples with no usable "
                  f"label logprobs")
    finally:
        if judge is not None:
            judge.shutdown()
