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
    return parser.parse_args()


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

    ok = (label_agreement >= args.label_agreement_threshold
          and correlation >= args.correlation_threshold)
    print("\n" + ("PASS -- reimplementation looks faithful; safe to score the other judges."
                  if ok else
                  "FAIL -- check the label token ids and the chat template before continuing."))
    sys.exit(0 if ok else 1)
