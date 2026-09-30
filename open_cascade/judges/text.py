"""Text-only LLM judge (the paper's setting)."""
import math
from typing import Dict, List, Optional

from model.prompts import (
    fewshot_example_prompt,
    fewshot_inst_prompt,
    fewshot_query_prompt,
    system_prompt,
)
from open_cascade.judges.base import (
    ASSISTANT_PREFIX,
    BaseJudge,
    LabelProbs,
    average_simulations,
    response_orderings,
)
from open_cascade.registry import JudgeConfig


class TextJudge(BaseJudge):
    """Text judge. Prompts are the authors' few-shot prompts from `model/prompts.py`.

    Args (in addition to BaseJudge's):
        released_mistral_compat: score the way the released Mistral judgements were made
            (generate 5 tokens and parse "[[A]]"). Only for validating against
            ./result/mistral-7b-instruct.*.jsonl.
    """

    def __init__(self, model_name: str, config: JudgeConfig, enable_prefix_caching: bool = True,
                 max_fewshot_examples: Optional[int] = None,
                 released_mistral_compat: bool = False):
        # set before super().__init__, which renders a probe prompt via _render
        self.released_mistral_compat = released_mistral_compat
        super().__init__(model_name, config, enable_prefix_caching, max_fewshot_examples)

    # prompt construction
    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.config.hf_name)

    def _render(self, user_content: str) -> str:
        """Render one user turn, open the assistant turn, seed it with "[["."""
        if any(model in self.config.hf_name for model in ("Mistral", "Mixtral")):
            messages = [{"role": "user", "content": user_content}]
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            return rendered if self.released_mistral_compat else rendered + ASSISTANT_PREFIX

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            head = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # some templates (e.g. Mistral v0.2) reject a system role
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]
            head = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return head + ASSISTANT_PREFIX

    def _probe_prompt(self) -> str:
        return self._render(
            fewshot_query_prompt.format(instruction="probe", assistant_a="a", assistant_b="b"))

    @staticmethod
    def format_user_prompt(instruction: str, assistant_a: str, assistant_b: str,
                           fewshot_examples: List[Dict]) -> str:
        """The few-shot user message: task description, K examples, then the query."""
        prompt = fewshot_inst_prompt

        for example in fewshot_examples:
            preferred_response = "[[A]]" if example["preferences"]["human"] == 1 else "[[B]]"
            prompt += "\n" + fewshot_example_prompt.format(
                instruction=example["instruction"],
                assistant_a=example["outputs"][0],
                assistant_b=example["outputs"][1],
                preferred_response=preferred_response,
            )

        prompt += "\n" + fewshot_query_prompt.format(
            instruction=instruction, assistant_a=assistant_a, assistant_b=assistant_b)
        return prompt

    def build_prompt(self, instruction: str, assistant_a: str, assistant_b: str,
                     fewshot_examples: List[Dict]) -> List[int]:
        """Token ids of the full rendered prompt, ending in "[["."""
        user_prompt = self.format_user_prompt(instruction, assistant_a, assistant_b,
                                              fewshot_examples)
        return self._encode(self._render(user_prompt))

    def build_request(self, sample: Dict, assistant_a: str, assistant_b: str,
                      fewshot_examples: List[Dict]) -> Dict:
        return {"prompt_token_ids": self.build_prompt(
            sample["instruction"], assistant_a, assistant_b, fewshot_examples)}

    # scoring
    def _generate(self, requests: List[Dict]):
        try:
            return super()._generate(requests)
        except TypeError:
            # older vLLM releases take prompt_token_ids as a keyword argument
            from vllm import SamplingParams

            sampling_params = SamplingParams(n=1, temperature=0, max_tokens=1, logprobs=20)
            return self.llm.generate(
                prompt_token_ids=[request["prompt_token_ids"] for request in requests],
                sampling_params=sampling_params, use_tqdm=False)

    def simulate_annotators_batch(self, samples: List[Dict],
                                  fewshot_examples_list: List[List[Dict]]) -> List[List[float]]:
        if self.released_mistral_compat:
            return self._simulate_annotators_released_compat(samples, fewshot_examples_list)
        return super().simulate_annotators_batch(samples, fewshot_examples_list)

    # released-Mistral compatibility path (validation only)
    def _simulate_annotators_released_compat(
            self, samples: List[Dict],
            fewshot_examples_list: List[List[Dict]]) -> List[List[float]]:
        """Reproduce the released scoring: one vLLM call per sample and ordering."""
        if self.max_fewshot_examples is not None:
            fewshot_examples_list = [
                examples[:self.max_fewshot_examples] for examples in fewshot_examples_list
            ]

        results = []
        for sample in samples:
            provenance, scored = [], []
            for ordering, (a, b) in enumerate(response_orderings(sample)):
                prompts = [
                    self._render(self.format_user_prompt(sample["instruction"], a, b, examples))
                    for examples in fewshot_examples_list
                ]
                for probs in self._score_released_compat_prompts(prompts):
                    provenance.append((0, ordering))
                    scored.append(probs)
            results.extend(average_simulations(1, provenance, scored))

        assert len(results) == len(samples)
        return results

    def _score_released_compat_prompts(self, prompts: List[str]) -> List[LabelProbs]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=5, logprobs=5)
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        scored: List[LabelProbs] = []
        for output in outputs:
            completion = output.outputs[0]
            generation = completion.text.strip()
            if generation not in ("[[A]]", "[[B]]"):
                scored.append(None)
                continue

            result_logprobs = {"A": -math.inf, "B": -math.inf}
            token_list = [self.tokenizer.decode(token_id) for token_id in completion.token_ids]
            for idx, token in enumerate(token_list):
                if idx > 0 and "[[" in token_list[idx - 1] and ("A" in token or "B" in token):
                    top_logprobs = completion.logprobs[idx]
                    for label, token_id in self.label_token_ids.items():
                        if token_id in top_logprobs:
                            result_logprobs[label] = top_logprobs[token_id].logprob
                    break

            if math.isinf(result_logprobs["A"]) and math.isinf(result_logprobs["B"]):
                scored.append(None)
                continue

            # raw exp(logprob), not renormalised -- matching the released files
            scored.append({label: math.exp(lp) for label, lp in result_logprobs.items()})

        return scored
