"""Vision-language judge: same scoring as the text judge, with an image per example."""
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from model.vlm_prompts import (
    fewshot_example_prompt,
    fewshot_inst_prompt,
    fewshot_query_prompt,
    system_prompt,
)
from open_cascade.judges.base import ASSISTANT_PREFIX, BaseJudge
from open_cascade.registry import JudgeConfig


@lru_cache(maxsize=512)
def _load_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class VLMJudge(BaseJudge):
    """VLM judge. Prompts are in `model/vlm_prompts.py`; each instance has an `image_path`.

    Every few-shot example adds one image to the prompt, so `max_fewshot_examples` is
    required here and defaults to 1 to stay within context.
    """

    def __init__(self, model_name: str, config: JudgeConfig, enable_prefix_caching: bool = True,
                 max_fewshot_examples: Optional[int] = 1):
        if max_fewshot_examples is None:
            raise ValueError("VLM judges need max_fewshot_examples: each example adds an image.")
        super().__init__(model_name, config, enable_prefix_caching, max_fewshot_examples)

    def _extra_llm_kwargs(self) -> Dict:
        # one image per few-shot example, plus the query image
        limit = self.config.limit_mm_per_prompt or {"image": self.max_fewshot_examples + 1}
        return {"limit_mm_per_prompt": limit}

    # prompt construction
    def _load_tokenizer(self):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.config.hf_name)
        return getattr(self.processor, "tokenizer", self.processor)

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

    def _probe_prompt(self) -> str:
        probe_image = Image.new("RGB", (1, 1), "white")
        content = [
            {"type": "text", "text": fewshot_inst_prompt},
            {"type": "image", "image": probe_image},
            {"type": "text", "text": fewshot_query_prompt.format(
                instruction="probe", assistant_a="a", assistant_b="b")},
        ]
        return self._render(content)

    @staticmethod
    def _sample_image(sample: Dict) -> Image.Image:
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(
                f"Missing image {image_path}. Run `prepare-data` from the repo root, "
                "or store image_path values relative to the current working directory."
            )
        return _load_image(str(image_path))

    def build_prompt(self, sample: Dict, assistant_a: str, assistant_b: str,
                     fewshot_examples: List[Dict]) -> Tuple[str, List[Image.Image]]:
        """Rendered prompt (ending in "[[") and its images, in the order they appear."""
        content: List[Dict] = [{"type": "text", "text": fewshot_inst_prompt}]
        images: List[Image.Image] = []

        for example in fewshot_examples:
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

    def build_request(self, sample: Dict, assistant_a: str, assistant_b: str,
                      fewshot_examples: List[Dict]) -> Dict:
        prompt, images = self.build_prompt(sample, assistant_a, assistant_b, fewshot_examples)
        return {"prompt": prompt, "multi_modal_data": {"image": images}}
