"""`validate` stage: check a re-scored judge against the judgements shipped in ./result/.

Run this on `mistral-7b-instruct` before spending GPU time on the larger judges. The repo
ships that judge's results on exactly these instances, so it directly tests whether the
prompting and scoring path in `open_cascade/judges/` is faithful.

Expect close but not identical results. Intentional differences:
  * teacher-forced "[[" and one decode step, vs. generate-5-tokens and string match
  * probabilities renormalised over {A, B}, vs. raw exp(logprob)
  * one BOS, vs. the double BOS the original chat-template path produced
  * add_generation_prompt=True on the chat template

None of these should move the predicted label or human agreement much. The exact confidence
scale can drift across vLLM versions, kernels, prompt batching, and whether probabilities
come from generation-time logprobs or teacher-forced next-token logprobs.
"""
from collections import Counter
from typing import Dict, List

import numpy as np

from open_cascade.config import ExperimentConfig
from open_cascade.data import read_jsonl, sample_key


def _truncate(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def _print_examples(title: str, indices, shared: List[str], released: Dict[str, Dict],
                    arrays: Dict[str, np.ndarray]) -> None:
    if len(indices) == 0:
        return

    print(f"\n{title}")
    print("-" * len(title))
    rel, rep = arrays["rel"], arrays["rep"]
    for rank, idx in enumerate(indices, start=1):
        sample = released[shared[idx]]
        print(f"[{rank}] index={idx} human={int(arrays['human'][idx]) + 1} "
              f"released_label={int(arrays['rel_label'][idx]) + 1} "
              f"reproduced_label={int(arrays['rep_label'][idx]) + 1}")
        print(f"    released_probs   : [{rel[idx, 0]:.6f}, {rel[idx, 1]:.6f}] "
              f"phat={arrays['rel_phat'][idx]:.6f}")
        print(f"    reproduced_probs : [{rep[idx, 0]:.6f}, {rep[idx, 1]:.6f}] "
              f"phat={arrays['rep_phat'][idx]:.6f}")
        print(f"    delta_phat       : "
              f"{abs(arrays['rel_phat'][idx] - arrays['rep_phat'][idx]):.6f}")
        print(f"    instruction      : {_truncate(sample['instruction'])}")
        print(f"    output_1         : {_truncate(sample['outputs'][0])}")
        print(f"    output_2         : {_truncate(sample['outputs'][1])}")


def validate(config: ExperimentConfig) -> bool:
    """Compare `{result_dir}/{judge}.{split}.jsonl` with `validation.released_file`."""
    if len(config.judges) != 1:
        raise ValueError("A validation config should list exactly one judge.")
    v = config.validation
    reproduced_file = config.result_file(config.judges[0], v.split)

    released = {sample_key(s): s for s in read_jsonl(v.released_file)}
    reproduced = {sample_key(s): s for s in read_jsonl(reproduced_file)}

    shared = sorted(set(released) & set(reproduced))
    if not shared:
        raise SystemExit("No overlapping instances -- are these the same judge and split?")

    print(f"released    {len(released):>5}  ({v.released_file})")
    print(f"reproduced  {len(reproduced):>5}  ({reproduced_file})")
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
    rel_human_agreement = float((rel_label == human).mean())
    rep_human_agreement = float((rep_label == human).mean())

    print(f"label agreement (released vs reproduced) : {label_agreement:.4f}")
    print(f"confidence correlation (Pearson)         : {correlation:.4f}")
    print(f"mean |change in phat|                    : {float(np.abs(rel_phat - rep_phat).mean()):.4f}")
    print(f"max  |change in phat|                    : {float(np.abs(rel_phat - rep_phat).max()):.4f}\n")
    print(f"human agreement, released                : {rel_human_agreement:.4f}")
    print(f"human agreement, reproduced              : {rep_human_agreement:.4f}")
    print(f"mean phat, released / reproduced         : {rel_phat.mean():.4f} / {rep_phat.mean():.4f}")

    if v.debug_examples > 0:
        arrays = dict(rel=rel, rep=rep, human=human, rel_label=rel_label, rep_label=rep_label,
                      rel_phat=rel_phat, rep_phat=rep_phat)
        worst = np.argsort(-np.abs(rel_phat - rep_phat))[:v.debug_examples]
        flips = np.flatnonzero(rel_label != rep_label)[:v.debug_examples]
        _print_examples("largest confidence shifts", worst, shared, released, arrays)
        _print_examples("first label flips", flips, shared, released, arrays)

    ok = (label_agreement >= v.label_agreement_threshold
          and abs(rel_human_agreement - rep_human_agreement) <= v.human_agreement_tolerance)
    if v.strict_confidence:
        ok = ok and correlation >= v.correlation_threshold

    if ok:
        print("\nPASS -- labels and human agreement look faithful enough to continue.")
        if correlation < v.correlation_threshold:
            print("Note: confidence correlation is low, so treat this as vLLM/logprob drift, "
                  "not byte-level reproduction of the released artifact.")
    else:
        print("\nFAIL -- labels or human agreement drifted too much. "
              "Inspect the debug examples above.")
    return ok
