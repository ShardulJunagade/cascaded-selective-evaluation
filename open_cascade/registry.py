"""Judge model configurations, text and vision-language.

Registering a new judge means adding one entry to `JUDGE_REGISTRY` and nothing else.
`modality` picks the judge class: "text" -> judges.text.TextJudge, "vlm" -> judges.vlm.VLMJudge.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

# The paper's cascade, and the open models substituted for the two API judges.
PAPER_TO_OPEN = {
    "mistral-7b-instruct": "mistral-7b-instruct",   # already open, unchanged
    "gpt-3.5-turbo": "qwen2.5-7b-instruct",
    "gpt-4-turbo": "qwen2.5-72b-instruct",
}

# Measured prompt lengths vary by chat template and tokenizer. Mistral v0.2 can exceed
# 4096 tokens on a few samples with the shipped N=K=3 few-shot examples once the full
# tokenizer template is applied.
# Capping max_model_len here rather than letting vLLM default to the model's full context
# (32k+) still shrinks the KV cache it must reserve, and avoids the common startup failure
# "The model's max seq len is larger than the maximum number of tokens that can be stored
# in the KV cache". Raise this if you increase K or move to a longer-context dataset.
TEXT_MAX_MODEL_LEN = 8192
# VLM prompts carry image tokens as well. Keep this aligned with the corpus
# preparation budget: a real Granite K=2 prompt reaches 17k tokens after its
# three images are expanded, so the previous 16k cap rejected valid records.
VLM_MAX_MODEL_LEN = 32768


@dataclass(frozen=True)
class JudgeConfig:
    hf_name: str
    modality: str = "text"                  # "text" or "vlm"

    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1           # number of GPUs the model is sharded over
    quantization: Optional[str] = None      # e.g. "awq", "gptq"
    max_model_len: Optional[int] = TEXT_MAX_MODEL_LEN
    gpu_memory_utilization: float = 0.90

    # VLM only. None = one image per few-shot example plus one for the query.
    limit_mm_per_prompt: Optional[Dict[str, int]] = None
    trust_remote_code: bool = False          # model/processor supplied by the HF repository

    extra_llm_kwargs: Dict = field(default_factory=dict)  # passed straight to vllm.LLM

    # Hosted judges (judges/api.py). When `provider` is set, no local model is loaded:
    # `hf_name` is the provider's model id, and the vLLM fields above are ignored.
    # `provider` is a key of judges.api.PROVIDERS ("openrouter", "gemini").
    # `api` overrides judges.api.DEFAULT_API_OPTIONS (confidence mode, max_tokens, ...).
    provider: Optional[str] = None
    api: Dict = field(default_factory=dict)


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

    # ------------------------- vision-language judges ------------------------ #
    "qwen2.5-vl-3b-instruct": JudgeConfig(
        hf_name="Qwen/Qwen2.5-VL-3B-Instruct",
        modality="vlm",
        max_model_len=VLM_MAX_MODEL_LEN,
    ),
    "qwen2.5-vl-7b-instruct": JudgeConfig(
        hf_name="Qwen/Qwen2.5-VL-7B-Instruct",
        modality="vlm",
        max_model_len=VLM_MAX_MODEL_LEN,
    ),
    "qwen2.5-vl-72b-instruct": JudgeConfig(
        hf_name="Qwen/Qwen2.5-VL-72B-Instruct",
        modality="vlm",
        max_model_len=VLM_MAX_MODEL_LEN,
        tensor_parallel_size=4,
    ),
    # -- multi-family VLM cascade: increasing parameter count, one GPU each -- #
    "granite-vision-3.3-2b": JudgeConfig(
        hf_name="ibm-granite/granite-vision-3.3-2b",
        modality="vlm",
        max_model_len=VLM_MAX_MODEL_LEN,
    ),
    "phi-3.5-vision-instruct": JudgeConfig(
        hf_name="microsoft/Phi-3.5-vision-instruct",
        modality="vlm",
        max_model_len=VLM_MAX_MODEL_LEN,
        trust_remote_code=True,
    ),
    "internvl3-8b": JudgeConfig(
        hf_name="OpenGVLab/InternVL3-8B",
        modality="vlm",
        max_model_len=VLM_MAX_MODEL_LEN,
        trust_remote_code=True,
    ),

    # ---------------------- hosted (API) judges, judges/api.py --------------------- #
    # Kimi K2.5 via OpenRouter (OPENROUTER_API_KEY). Text + image input.
    # Confidence from the A/B token logprobs. Pinned to Novita, measured 2026-11-07 with
    # `test_api_judge.py --probe`: only Novita and Venice return logprobs for this model,
    # and Venice ignores every way of disabling thinking (it reasons in the answer text
    # and runs out of tokens before the verdict). Novita honours reasoning.enabled=false
    # and answers "[[A]]" in 4 tokens. No fallback, so a Novita outage fails loudly rather
    # than silently routing to Venice. Re-run --probe if this starts failing.
    "kimi-k2.5": JudgeConfig(
        hf_name="moonshotai/kimi-k2.5",
        modality="vlm",
        provider="openrouter",
        api={
            "confidence": "logprobs",
            "max_tokens": 16,
            "extra_body": {"reasoning": {"enabled": False},
                           "provider": {"order": ["novita"], "allow_fallbacks": False,
                                        "require_parameters": True}},
        },
    ),
    # Fallback if no logprob-capable provider is up: confidence = vote share across
    # the 2N simulations (each a hard verdict), so it is coarser.
    "kimi-k2.5-votes": JudgeConfig(
        hf_name="moonshotai/kimi-k2.5",
        modality="vlm",
        provider="openrouter",
        api={
            "confidence": "votes",
            "max_tokens": 16,
            "extra_body": {"reasoning": {"enabled": False}},
        },
    ),
    # Gemini direct from Google AI Studio (GEMINI_API_KEY), through its OpenAI-compatible
    # endpoint. Gemini does not expose logprobs there or on OpenRouter, so it uses votes.
    # Gemini thinks before answering and thinking tokens count against max_tokens, so
    # max_tokens is kept generous.
    "gemini-flash": JudgeConfig(
        hf_name="gemini-flash-latest",
        modality="vlm",
        provider="gemini",
        api={
            "confidence": "votes",
            "max_tokens": 2048,
            "extra_body": {"reasoning_effort": "low"},
        },
    ),
    # The same model routed through OpenRouter, if you have credit there but no Google key.
    "gemini-flash-openrouter": JudgeConfig(
        hf_name="~google/gemini-flash-latest",
        modality="vlm",
        provider="openrouter",
        api={
            "confidence": "votes",
            "max_tokens": 2048,
            "extra_body": {"reasoning": {"effort": "low"}},
        },
    ),
}


def resolve_judge(model_name: str) -> JudgeConfig:
    if model_name not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge '{model_name}'. Known judges: {sorted(JUDGE_REGISTRY)}. "
            f"Add an entry to open_cascade/registry.py to register a new one."
        )
    return JUDGE_REGISTRY[model_name]
