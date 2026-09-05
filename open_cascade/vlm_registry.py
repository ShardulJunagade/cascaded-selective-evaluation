"""Vision-language judge model configurations."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_OPEN_VLM_CASCADE: List[str] = [
    "qwen2.5-vl-3b-instruct",
    "qwen2.5-vl-7b-instruct",
]

DEFAULT_VLM_MAX_MODEL_LEN = 16384


@dataclass(frozen=True)
class VLMJudgeConfig:
    hf_name: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None
    max_model_len: Optional[int] = DEFAULT_VLM_MAX_MODEL_LEN
    gpu_memory_utilization: float = 0.90
    limit_mm_per_prompt: Dict[str, int] = field(default_factory=lambda: {"image": 2})
    extra_llm_kwargs: Dict = field(default_factory=dict)


VLM_JUDGE_REGISTRY: Dict[str, VLMJudgeConfig] = {
    "qwen2.5-vl-3b-instruct": VLMJudgeConfig(
        hf_name="Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    "qwen2.5-vl-7b-instruct": VLMJudgeConfig(
        hf_name="Qwen/Qwen2.5-VL-7B-Instruct",
    ),
    "qwen2.5-vl-72b-instruct": VLMJudgeConfig(
        hf_name="Qwen/Qwen2.5-VL-72B-Instruct",
        tensor_parallel_size=4,
    ),
}


def resolve_vlm_judge(model_name: str) -> VLMJudgeConfig:
    if model_name not in VLM_JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown VLM judge '{model_name}'. Known judges: {sorted(VLM_JUDGE_REGISTRY)}. "
            "Add an entry to open_cascade/vlm_registry.py to register a new one."
        )
    return VLM_JUDGE_REGISTRY[model_name]
