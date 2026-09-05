"""Rebuild the raw dataset and splits from the released judgement files.

The repo ships judgements (`./result/*.jsonl`) but not the underlying data, so there is
nothing to feed a new judge. Each released file carries the full instance next to its
`probs`, so stripping `probs` recovers it.

This preserves the *original* calibration/test split, which is what makes a newly scored
open judge directly comparable against the released gpt-4-turbo / gpt-3.5-turbo results.

Usage:
    python reconstruct_dataset.py
"""
from argparse import ArgumentParser
from pathlib import Path

from open_cascade.data import read_jsonl, sample_key, strip_judgement, write_jsonl

SPLITS = ("calibration", "test")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--reference_model", default="gpt-4-turbo",
                        help="Released judge whose result files supply the instances.")
    parser.add_argument("--verify_against", nargs="*",
                        default=["gpt-3.5-turbo", "mistral-7b-instruct"],
                        help="Other released judges to cross-check instance alignment against.")
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument("--out_dir", default="./data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_dir, out_dir = Path(args.result_dir), Path(args.out_dir)

    splits = {
        split: [strip_judgement(s)
                for s in read_jsonl(result_dir / f"{args.reference_model}.{split}.jsonl")]
        for split in SPLITS
    }

    keys = {split: {sample_key(s) for s in samples} for split, samples in splits.items()}

    overlap = keys["calibration"] & keys["test"]
    if overlap:
        raise SystemExit(f"calibration and test overlap by {len(overlap)} instances")

    for other in args.verify_against:
        for split in SPLITS:
            path = result_dir / f"{other}.{split}.jsonl"
            if not path.exists():
                print(f"  skipping cross-check: {path} not found")
                continue
            other_keys = {sample_key(s) for s in read_jsonl(path)}
            if other_keys != keys[split]:
                raise SystemExit(
                    f"{other}.{split} covers different instances than "
                    f"{args.reference_model}.{split} ({len(other_keys ^ keys[split])} differ). "
                    f"The splits are not aligned; comparisons would be invalid."
                )
        print(f"  cross-checked against {other}: identical instances in both splits")

    for split in SPLITS:
        path = out_dir / "split" / f"{split}.jsonl"
        write_jsonl(splits[split], path)
        print(f"  wrote {len(splits[split]):>5} -> {path}")

    pooled = splits["calibration"] + splits["test"]
    write_jsonl(pooled, out_dir / "preprocessed" / "data.jsonl")
    print(f"  wrote {len(pooled):>5} -> {out_dir / 'preprocessed' / 'data.jsonl'}")

    print(f"\nScore an open judge on the same split with:")
    print(f"  python run_open_judge.py --model_name=qwen2.5-7b-instruct "
          f"--in_filename={out_dir / 'split' / 'calibration.jsonl'}")
