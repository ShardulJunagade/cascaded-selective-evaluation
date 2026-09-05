"""Simulated Annotators for open-weight vision-language judges served with vLLM."""
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from transformers import AutoProcessor

from model.vlm_prompts import (
    fewshot_example_prompt,
    fewshot_inst_prompt,
    fewshot_query_prompt,
    system_prompt,
)
from open_cascade.vlm_registry import resolve_vlm_judge

LABELS = ("A", "B")
ASSISTANT_PREFIX = "[["
ORDERING_CONVERTERS = ({"A": 1, "B": 2}, {"B": 1, "A": 2})


@lru_cache(maxsize=512)
def _load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class OpenVLMJudge:
    """An open-weight VLM judge scored via teacher-forced A/B label logprobs."""

    def __init__(self, model_name: str, enable_prefix_caching: bool = True,
                 max_fewshot_examples: int = 1):
        logging.getLogger("vllm").setLevel(logging.WARNING)

        from vllm import LLM

        self.model_name = model_name
        self.config = resolve_vlm_judge(model_name)
        self.max_fewshot_examples = max_fewshot_examples

        llm_kwargs = dict(
            model=self.config.hf_name,
            seed=42,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt=self.config.limit_mm_per_prompt,
            **self.config.extra_llm_kwargs,
        )
        if self.config.quantization is not None:
            llm_kwargs["quantization"] = self.config.quantization
        if self.config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_len

        self.llm = LLM(**llm_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.config.hf_name)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.label_token_ids = self._resolve_label_token_ids()

    def shutdown(self) -> None:
        """Best-effort vLLM cleanup before Python's multiprocessing atexit hook runs."""
        llm_engine = getattr(self.llm, "llm_engine", None)
        for target in (llm_engine, getattr(llm_engine, "engine_core", None), self.llm):
            shutdown = getattr(target, "shutdown", None)
            if shutdown is None:
                continue
            try:
                shutdown()
                break
            except TypeError:
                shutdown(timeout=0)
                break

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _render(self, content: List[Dict]) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        try:
            head = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": system_prompt}] + content,
                }
            ]
            head = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return head + ASSISTANT_PREFIX

    def _resolve_label_token_ids(self) -> Dict[str, int]:
        probe_image = Image.new("RGB", (1, 1), "white")
        content = [
            {"type": "text", "text": fewshot_inst_prompt},
            {"type": "image", "image": probe_image},
            {"type": "text", "text": fewshot_query_prompt.format(
                instruction="probe", assistant_a="a", assistant_b="b")},
        ]
        probe = self._render(content)
        base = self._encode(probe)

        label_token_ids = {}
        for label in LABELS:
            full = self._encode(probe + label)
            if full[: len(base)] != base or len(full) != len(base) + 1:
                raise RuntimeError(
                    f"Tokenizer for '{self.config.hf_name}' does not put '{label}' on a clean "
                    f"token boundary after '{ASSISTANT_PREFIX}'."
                )
            label_token_ids[label] = full[-1]

        if label_token_ids["A"] == label_token_ids["B"]:
            raise RuntimeError(f"'A' and 'B' share a token id for {self.config.hf_name}.")
        return label_token_ids

    def _sample_image(self, sample: Dict) -> Image.Image:
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image {image_path}. Run prepare_vlm_dataset.py from the repo root, "
                "or store image_path values relative to the current working directory."
            )
        return _load_image(str(image_path))

    def build_prompt(self, sample: Dict, assistant_a: str, assistant_b: str,
                     fewshot_examples: List[Dict]) -> Tuple[str, List[Image.Image]]:
        content: List[Dict] = [{"type": "text", "text": fewshot_inst_prompt}]
        images: List[Image.Image] = []

        for example in fewshot_examples[:self.max_fewshot_examples]:
            preferred_response = "[[A]]" if example["preferences"]["human"] == 1 else "[[B]]"
            image = self._sample_image(example)
            images.append(image)
            content.extend([
                {"type": "image", "image": image},
                {"type": "text", "text": fewshot_example_prompt.format(
                    instruction=example["instruction"],
                    assistant_a=example["outputs"][0],
                    assistant_b=example["outputs"][1],
                    preferred_response=preferred_response,
                )},
            ])

        image = self._sample_image(sample)
        images.append(image)
        content.extend([
            {"type": "image", "image": image},
            {"type": "text", "text": fewshot_query_prompt.format(
                instruction=sample["instruction"],
                assistant_a=assistant_a,
                assistant_b=assistant_b,
            )},
        ])

        return self._render(content), images

    def _generate(self, requests: List[Dict]):
        from vllm import SamplingParams

        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=1, logprobs=20)
        return self.llm.generate(requests, sampling_params, use_tqdm=False)

    def score_prompts(self, requests: List[Dict]) -> List[Optional[Dict[str, float]]]:
        scored: List[Optional[Dict[str, float]]] = []

        for output in self._generate(requests):
            step_logprobs = output.outputs[0].logprobs[0]
            logprobs = {
                label: step_logprobs[token_id].logprob
                for label, token_id in self.label_token_ids.items()
                if token_id in step_logprobs
            }
            if not logprobs:
                scored.append(None)
                continue
            if len(logprobs) == 1:
                floor = min(lp.logprob for lp in step_logprobs.values()) - 1.0
                missing = next(label for label in LABELS if label not in logprobs)
                logprobs[missing] = floor

            offset = max(logprobs.values())
            unnormalised = {label: math.exp(lp - offset) for label, lp in logprobs.items()}
            total = sum(unnormalised.values())
            scored.append({label: value / total for label, value in unnormalised.items()})

        return scored

    def simulate_annotators_batch(self, samples: List[Dict],
                                  fewshot_examples_list: List[List[Dict]]) -> List[List[float]]:
        if self.max_fewshot_examples < 0:
            raise ValueError("--max_fewshot_examples must be non-negative")

        requests: List[Dict] = []
        provenance: List[tuple] = []

        for fewshot_examples in fewshot_examples_list:
            for sample_idx, sample in enumerate(samples):
                first, second = sample["outputs"][0], sample["outputs"][1]
                for ordering, (a, b) in enumerate(((first, second), (second, first))):
                    prompt, images = self.build_prompt(sample, a, b, fewshot_examples)
                    requests.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": images},
                    })
                    provenance.append((sample_idx, ordering))

        scored = self.score_prompts(requests)

        per_sample: List[List[Dict[int, float]]] = [[] for _ in samples]
        for (sample_idx, ordering), probs in zip(provenance, scored):
            if probs is None:
                continue
            converter = ORDERING_CONVERTERS[ordering]
            per_sample[sample_idx].append({converter[label]: p for label, p in probs.items()})

        results = []
        for prob_dicts in per_sample:
            if not prob_dicts:
                results.append([])
                continue
            results.append([
                float(np.mean([d[1] for d in prob_dicts])),
                float(np.mean([d[2] for d in prob_dicts])),
            ])

        assert len(results) == len(samples)
        return results
