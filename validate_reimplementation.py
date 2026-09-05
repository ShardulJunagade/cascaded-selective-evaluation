"""Check a re-scored judge against the judgements shipped in ./result/.

Run this on `mistral-7b-instruct` before spending GPU time on the larger judges. The repo
ships that judge's results on exactly these instances, so it directly tests whether the
prompting and scoring path in `open_cascade/judge.py` is faithful.

Expect close but not identical results. Intentional differences:
  * teacher-forced "[[" and one decode step, vs. generate-5-tokens and string match
  * probabilities renormalised over {A, B}, vs. raw exp(logprob)
  * one BOS, vs. the double BOS the original chat-template path produced
  * add_generation_prompt=True on the chat template

None of these should move the predicted label much. Label agreement below ~0.95 or
confidence correlation below ~0.90 means something is genuinely wrong -- most likely the
label token ids or the chat template.

Usage:
    python validate_reimplementation.py --reproduced=./result/_repro.mistral....jsonl
"""
import sys
from argparse import ArgumentParser
from collections import Counter

import numpy as np

from open_cascade.data import read_jsonl, sample_key


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--released", default="./result/mistral-7b-instruct.calibration.jsonl")
    parser.add_argument("--reproduced", required=True,
                        help="run_open_judge.py output for the same judge and split")
    parser.add_argument("--label_agreement_threshold", type=float, default=0.95)
    parser.add_argument("--correlation_threshold", type=float, default=0.90)
    parser.add_argument("--debug_examples", type=int, default=8,
                        help="Print the largest confidence shifts and label flips")
    return parser.parse_args()


def truncate(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def print_debug_examples(title, indices, shared, released, reproduced, rel, rep,
                         rel_label, rep_label, human, rel_phat, rep_phat):
    if len(indices) == 0:
        return

    print(f"\n{title}")
    print("-" * len(title))
    for rank, idx in enumerate(indices, start=1):
        key = shared[idx]
        sample = released[key]
        print(f"[{rank}] index={idx} human={int(human[idx]) + 1} "
              f"released_label={int(rel_label[idx]) + 1} reproduced_label={int(rep_label[idx]) + 1}")
        print(f"    released_probs   : [{rel[idx, 0]:.6f}, {rel[idx, 1]:.6f}] "
              f"phat={rel_phat[idx]:.6f}")
        print(f"    reproduced_probs : [{rep[idx, 0]:.6f}, {rep[idx, 1]:.6f}] "
              f"phat={rep_phat[idx]:.6f}")
        print(f"    delta_phat       : {abs(rel_phat[idx] - rep_phat[idx]):.6f}")
        print(f"    instruction      : {truncate(sample['instruction'])}")
        print(f"    output_1         : {truncate(sample['outputs'][0])}")
        print(f"    output_2         : {truncate(sample['outputs'][1])}")


if __name__ == "__main__":
    args = parse_args()

    released = {sample_key(s): s for s in read_jsonl(args.released)}
    reproduced = {sample_key(s): s for s in read_jsonl(args.reproduced)}

    shared = sorted(set(released) & set(reproduced))
    if not shared:
        raise SystemExit("No overlapping instances -- are these the same judge and split?")

    print(f"released    {len(released):>5}  ({args.released})")
    print(f"reproduced  {len(reproduced):>5}  ({args.reproduced})")
    print(f"overlap     {len(shared):>5}\n")

    rel = np.array([released[k]["probs"] for k in shared])
    rep = np.array([reproduced[k]["probs"] for k in shared])
    human = np.array([
        Counter(released[k]["preferences"].values()).most_common(1)[0][0] for k in shared
    ]) - 1

    rel_label, rep_label = rel.argmax(-1), rep.argmax(-1)
    rel_phat, rep_phat = rel.max(-1), rep.max(-1)

    label_agreement = float((rel_label == rep_label).mean())
    correlation = float(np.corrcoef(rel_phat, rep_phat)[0, 1])

    print(f"label agreement (released vs reproduced) : {label_agreement:.4f}")
    print(f"confidence correlation (Pearson)         : {correlation:.4f}")
    print(f"mean |change in phat|                    : {float(np.abs(rel_phat - rep_phat).mean()):.4f}")
    print(f"max  |change in phat|                    : {float(np.abs(rel_phat - rep_phat).max()):.4f}\n")

    print(f"human agreement, released                : {float((rel_label == human).mean()):.4f}")
    print(f"human agreement, reproduced              : {float((rep_label == human).mean()):.4f}")
    print(f"mean phat, released / reproduced         : {rel_phat.mean():.4f} / {rep_phat.mean():.4f}")

    if args.debug_examples > 0:
        deltas = np.abs(rel_phat - rep_phat)
        worst = np.argsort(-deltas)[:args.debug_examples]
        flips = np.flatnonzero(rel_label != rep_label)[:args.debug_examples]
        print_debug_examples(
            "largest confidence shifts",
            worst,
            shared,
            released,
            reproduced,
            rel,
            rep,
            rel_label,
            rep_label,
            human,
            rel_phat,
            rep_phat,
        )
        print_debug_examples(
            "first label flips",
            flips,
            shared,
            released,
            reproduced,
            rel,
            rep,
            rel_label,
            rep_label,
            human,
            rel_phat,
            rep_phat,
        )

    ok = (label_agreement >= args.label_agreement_threshold
          and correlation >= args.correlation_threshold)
    print("\n" + ("PASS -- reimplementation looks faithful; safe to score the other judges."
                  if ok else
                  "FAIL -- check the label token ids and the chat template before continuing."))
    sys.exit(0 if ok else 1)
