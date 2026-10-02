"""Source adapters, registered by name."""

from typing import Dict, Type

from prepare_data.adapters.base import Adapter
from prepare_data.adapters.judge_anything import JudgeAnythingAdapter
from prepare_data.adapters.mm_rlhf import MmRlhfAdapter
from prepare_data.adapters.rlhf_v import RlhfVAdapter
from prepare_data.adapters.visionarena import VisionArenaAdapter
from prepare_data.adapters.visit_bench import VisitBenchAdapter

ADAPTERS: Dict[str, Type[Adapter]] = {
    "rlhf_v": RlhfVAdapter,
    "visit_bench": VisitBenchAdapter,
    "judge_anything": JudgeAnythingAdapter,
    "mm_rlhf": MmRlhfAdapter,
    "visionarena": VisionArenaAdapter,
}

__all__ = ["Adapter", "ADAPTERS"]
