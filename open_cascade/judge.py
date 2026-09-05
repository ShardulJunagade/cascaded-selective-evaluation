"""Simulated Annotators (paper S2.2) for open-weight judges, served with vLLM.

Confidence is the agreement between N simulated annotators, each induced by K-shot
prompting with a different annotator's demonstrations:

    c_LM(x) = max_y (1/N) sum_j p_LM(y | x ; (x_1j, y_1j) ... (x_Kj, y_Kj))

Each simulation is additionally run in both response orderings, so a sample costs 2N
forward passes. The A/B swap is the paper's position-bias control, folded into the ensemble.

Differences from `model/vllm_model.py` (the authors' implementation), all of which exist to
make the method work across tokenizers rather than just Mistral's:

  * Label token ids are derived from the tokenizer instead of hardcoded. Deriving them
    naively is a trap: Mistral-7B-v0.2 encodes a bare "A" as 330 (the space-prefixed
    variant) while the token following "[[" is 28741. Reading logprobs for 330 returns -inf
    on every sample and yields an empty output file with no error raised anywhere.
  * `add_generation_prompt=True` on the chat template. Harmless to omit for Mistral, whose
    template appends [/INST] regardless, but fatal for Qwen: its ChatML prompt would end at
    <|im_end|> with no <|im_start|>assistant, leaving the model outside answering position.
  * The verdict is teacher-forced rather than generated. The assistant turn is opened with
    "[[" and the distribution over the single next token is read directly. The original
    generates 5 tokens, requires the text to equal "[[A]]" exactly, then hunts for the
    position after "[[" -- and drops the sample on any deviation. Weaker judges deviate
    often, and the loss is silent.
  * Prompts are built once as token ids, giving exactly one BOS. Re-tokenizing an
    already-templated string (as the original does) emits a second one for Mistral.
  * Probabilities are renormalised over {A, B}. The released values are raw exp(logprob),
    which is nearly identical in practice -- they sum to 1.000 at the median and 0.964 at
    worst, since A/B absorb essentially all mass in this position.
"""
import logging
import math
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer

from model.prompts import (
    fewshot_example_prompt,
    fewshot_inst_prompt,
    fewshot_query_prompt,
    system_prompt,
)
from open_cascade.registry import resolve_judge

LABELS = ("A", "B")
ASSISTANT_PREFIX = "[["

# label -> preference value, for each of the two response orderings
ORDERING_CONVERTERS = ({"A": 1, "B": 2}, {"B": 1, "A": 2})


class OpenJudge:
    """An open-weight LLM judge scored via teacher-forced label logprobs."""

    def __init__(self, model_name: str, enable_prefix_caching: bool = True,
                 released_mistral_compat: bool = False):
        logging.getLogger("vllm").setLevel(logging.WARNING)

        from vllm import LLM  # imported here so the module is importable without a GPU

        self.model_name = model_name
        self.config = resolve_judge(model_name)
        self.released_mistral_compat = released_mistral_compat

        llm_kwargs = dict(
            model=self.config.hf_name,
            seed=42,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            # There are only N distinct few-shot prefixes in an entire run, and the prefix is
            # ~75-85% of every prompt, so this is roughly a 4-5x saving.
            enable_prefix_caching=enable_prefix_caching,
            **self.config.extra_llm_kwargs,
        )
        if self.config.quantization is not None:
            llm_kwargs["quantization"] = self.config.quantization
        if self.config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_len

        self.llm = LLM(**llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.hf_name)
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

    # ------------------------------------------------------------------ #
    # prompt construction
    # ------------------------------------------------------------------ #
    def _render(self, user_content: str) -> str:
        """Render one user turn, open the assistant turn, seed it with "[[".."""
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

    def _encode(self, text: str) -> List[int]:
        # add_special_tokens=False: apply_chat_template already emits BOS as literal text
        # where the template calls for it, so letting the tokenizer add another doubles it.
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _resolve_label_token_ids(self) -> Dict[str, int]:
        """Ids of "A"/"B" in the position they actually occur -- immediately after "[[".

        Derived by diffing against the real prompt prefix, with the token boundary asserted,
        so that a tokenizer which behaves unexpectedly fails loudly at construction instead
        of silently producing no output.
        """
        probe = self._render(
            fewshot_query_prompt.format(instruction="probe", assistant_a="a", assistant_b="b"))
        base = self._encode(probe)

        label_token_ids = {}
        for label in LABELS:
            full = self._encode(probe + label)
            if full[: len(base)] != base or len(full) != len(base) + 1:
                raise RuntimeError(
                    f"Tokenizer for '{self.config.hf_name}' does not put '{label}' on a clean "
                    f"token boundary after '{ASSISTANT_PREFIX}' (prefix {len(base)} tokens, "
                    f"full {len(full)}). Teacher-forced scoring needs a stable boundary."
                )
            label_token_ids[label] = full[-1]

        if label_token_ids["A"] == label_token_ids["B"]:
            raise RuntimeError(f"'A' and 'B' share a token id for {self.config.hf_name}.")

        return label_token_ids

    def build_prompt(self, instruction: str, assistant_a: str, assistant_b: str,
                     fewshot_examples: List[Dict]) -> List[int]:
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

        return self._encode(self._render(prompt))

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #
    def _generate_released_compat(self, prompts: List[str]):
        from vllm import SamplingParams

        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=5, logprobs=5)
        return self.llm.generate(prompts, sampling_params, use_tqdm=False)

    def _generate(self, token_id_lists: List[List[int]]):
        from vllm import SamplingParams

        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=1, logprobs=20)
        try:
            return self.llm.generate(
                [{"prompt_token_ids": ids} for ids in token_id_lists],
                sampling_params, use_tqdm=False)
        except TypeError:
            # older vLLM releases take prompt_token_ids as a keyword argument
            return self.llm.generate(
                prompt_token_ids=token_id_lists,
                sampling_params=sampling_params, use_tqdm=False)

    def score_prompts(self, token_id_lists: List[List[int]]) -> List[Optional[Dict[str, float]]]:
        """P(A), P(B) per prompt, renormalised over the two labels. None if neither ranked."""
        scored: List[Optional[Dict[str, float]]] = []

        for output in self._generate(token_id_lists):
            step_logprobs = output.outputs[0].logprobs[0]

            logprobs = {
                label: step_logprobs[token_id].logprob
                for label, token_id in self.label_token_ids.items()
                if token_id in step_logprobs
            }

            if not logprobs:
                # neither label reached the top-20: the judge is answering off-format
                scored.append(None)
                continue
            if len(logprobs) == 1:
                # floor the absent label just below the least likely returned token
                floor = min(lp.logprob for lp in step_logprobs.values()) - 1.0
                missing = next(label for label in LABELS if label not in logprobs)
                logprobs[missing] = floor

            offset = max(logprobs.values())
            unnormalised = {label: math.exp(lp - offset) for label, lp in logprobs.items()}
            total = sum(unnormalised.values())
            scored.append({label: value / total for label, value in unnormalised.items()})

        return scored

    def build_released_compat_prompt(self, instruction: str, assistant_a: str,
                                     assistant_b: str, fewshot_examples: List[Dict]) -> str:
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

        return self._render(prompt)

    def score_released_compat_prompts(self, prompts: List[str]) -> List[Optional[Dict[str, float]]]:
        """Reproduce the released Mistral scoring path for validation only."""
        scored: List[Optional[Dict[str, float]]] = []

        for output in self._generate_released_compat(prompts):
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

            scored.append({
                label: math.exp(logprob)
                for label, logprob in result_logprobs.items()
            })

        return scored

    def simulate_annotators(self, sample: Dict,
                            fewshot_examples_list: List[List[Dict]]) -> List[float]:
        """Confidence for a single sample. Same signature as the authors' implementation."""
        return self.simulate_annotators_batch([sample], fewshot_examples_list)[0]

    def simulate_annotators_batch(self, samples: List[Dict],
                                  fewshot_examples_list: List[List[Dict]]) -> List[List[float]]:
        """Score many samples in one vLLM call.

        Prompts are emitted grouped by annotator so consecutive prompts share a few-shot
        prefix, which is what makes prefix caching effective.

        Returns `[P(preference=1), P(preference=2)]` per sample, or `[]` if every simulation
        for that sample failed to produce a usable label.
        """
        if self.released_mistral_compat:
            return self.simulate_annotators_batch_released_compat(samples, fewshot_examples_list)

        token_id_lists: List[List[int]] = []
        provenance: List[tuple] = []  # (sample index, ordering index)

        for fewshot_examples in fewshot_examples_list:
            for sample_idx, sample in enumerate(samples):
                first, second = sample["outputs"][0], sample["outputs"][1]
                for ordering, (a, b) in enumerate(((first, second), (second, first))):
                    token_id_lists.append(
                        self.build_prompt(sample["instruction"], a, b, fewshot_examples))
                    provenance.append((sample_idx, ordering))

        scored = self.score_prompts(token_id_lists)

        # map each simulation back into preference-label space, then average per sample
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

    def simulate_annotators_batch_released_compat(
            self, samples: List[Dict],
            fewshot_examples_list: List[List[Dict]]) -> List[List[float]]:
        results = []
        for sample in samples:
            first, second = sample["outputs"][0], sample["outputs"][1]
            predictive_probs: List[Dict[int, float]] = []

            for ordering, (a, b) in enumerate(((first, second), (second, first))):
                prompts = [
                    self.build_released_compat_prompt(
                        sample["instruction"], a, b, fewshot_examples)
                    for fewshot_examples in fewshot_examples_list
                ]
                scored = self.score_released_compat_prompts(prompts)
                converter = ORDERING_CONVERTERS[ordering]
                for probs in scored:
                    if probs is None:
                        continue
                    predictive_probs.append({converter[label]: p for label, p in probs.items()})

            if not predictive_probs:
                results.append([])
                continue
            results.append([
                float(np.mean([d[1] for d in predictive_probs])),
                float(np.mean([d[2] for d in predictive_probs])),
            ])

        assert len(results) == len(samples)
        return results
