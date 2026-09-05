"""Judge model configurations.

Registering a new judge should mean adding one entry here and nothing else.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The paper's cascade, and the open models substituted for the two API judges.
PAPER_TO_OPEN = {
    "mistral-7b-instruct": "mistral-7b-instruct",   # already open, unchanged
    "gpt-3.5-turbo": "qwen2.5-7b-instruct",
    "gpt-4-turbo": "qwen2.5-72b-instruct",
}

# Default open cascade, weakest judge first.
DEFAULT_OPEN_CASCADE: List[str] = [
    "mistral-7b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-72b-instruct",
]


# Measured worst-case prompt over the full 5218-instance dataset with the shipped
# N=K=3 few-shot examples is 2913 tokens (median 1752, p99 2474), plus 1 generated token.
# Capping max_model_len here rather than letting vLLM default to the model's full context
# (32k+) shrinks the KV cache it must reserve by ~8x, and avoids the common startup failure
# "The model's max seq len is larger than the maximum number of tokens that can be stored
# in the KV cache". Raise this if you increase K or move to a longer-context dataset.
DEFAULT_MAX_MODEL_LEN = 4096


@dataclass(frozen=True)
class JudgeConfig:
    hf_name: str

    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None  # e.g. "awq", "gptq"
    max_model_len: Optional[int] = DEFAULT_MAX_MODEL_LEN
    gpu_memory_utilization: float = 0.90

    extra_llm_kwargs: Dict = field(default_factory=dict)


JUDGE_REGISTRY: Dict[str, JudgeConfig] = {
    # -- the paper's open judge; also the correctness anchor against ./result/*.jsonl -- #
    # dtype is "half" (not bfloat16) to match the setting the released judgements were
    # generated with, so the validation comparison stays apples-to-apples.
    "mistral-7b-instruct": JudgeConfig(
        hf_name="mistralai/Mistral-7B-Instruct-v0.2",
        dtype="half",
    ),
    # -- open replacement for gpt-3.5-turbo -- #
    "qwen2.5-7b-instruct": JudgeConfig(
        hf_name="Qwen/Qwen2.5-7B-Instruct",
    ),
    # -- open replacement for gpt-4-turbo -- #
    "qwen2.5-72b-instruct": JudgeConfig(
        hf_name="Qwen/Qwen2.5-72B-Instruct",
        tensor_parallel_size=4,
    ),

    # -- extra tiers for the judge-composition ablation (paper Table 6) -- #
    "qwen2.5-1.5b-instruct": JudgeConfig(hf_name="Qwen/Qwen2.5-1.5B-Instruct"),
    "qwen2.5-3b-instruct": JudgeConfig(hf_name="Qwen/Qwen2.5-3B-Instruct"),
    "qwen2.5-32b-instruct": JudgeConfig(
        hf_name="Qwen/Qwen2.5-32B-Instruct",
        tensor_parallel_size=2,
    ),
    # the paper's own "weaker cascades" mid judge (S3.6)
    "mixtral-8x7b-instruct": JudgeConfig(
        hf_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        dtype="half",
        tensor_parallel_size=2,
    ),
}


def resolve_judge(model_name: str) -> JudgeConfig:
    if model_name not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge '{model_name}'. Known judges: {sorted(JUDGE_REGISTRY)}. "
            f"Add an entry to open_cascade/registry.py to register a new one."
        )
    return JUDGE_REGISTRY[model_name]
