"""Judges: the models that produce `probs` for each instance.

    base.py   BaseJudge -- vLLM setup, label-token lookup, scoring, A/B-swap averaging
    text.py   TextJudge -- text prompts (the paper's setting)
    vlm.py    VLMJudge  -- text + image prompts

To add a judge *model*, add an entry to open_cascade/registry.py.
To add a new *kind* of judge (e.g. a different prompt format or modality), subclass
BaseJudge, implement its three hooks, and add the class to JUDGE_CLASSES below.
"""
from open_cascade.judges.base import BaseJudge
from open_cascade.judges.text import TextJudge
from open_cascade.judges.vlm import VLMJudge
from open_cascade.registry import resolve_judge

# JudgeConfig.modality -> judge class
JUDGE_CLASSES = {
    "text": TextJudge,
    "vlm": VLMJudge,
}


def build_judge(model_name: str, **kwargs) -> BaseJudge:
    """Load the judge registered as `model_name`. kwargs go to the judge class."""
    config = resolve_judge(model_name)
    if config.modality not in JUDGE_CLASSES:
        raise ValueError(f"Judge '{model_name}' has unknown modality '{config.modality}'. "
                         f"Known: {sorted(JUDGE_CLASSES)}")
    return JUDGE_CLASSES[config.modality](model_name, config, **kwargs)
