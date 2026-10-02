"""Simulated Annotators (paper S2.2) for open-weight judges, served with vLLM.

Confidence is the agreement between N simulated annotators, each induced by K-shot
prompting with a different annotator's demonstrations:

    c_LM(x) = max_y (1/N) sum_j p_LM(y | x ; (x_1j, y_1j) ... (x_Kj, y_Kj))

Each simulation is additionally run in both response orderings, so a sample costs 2N
forward passes. The A/B swap is the paper's position-bias control, folded into the ensemble.

`BaseJudge` holds everything that does not depend on the input modality: loading the model,
finding the label token ids, reading the verdict distribution and averaging it over
simulations. A subclass only decides how a prompt is built:

    _load_tokenizer()   the tokenizer (or processor) for the model
    _probe_prompt()     any rendered prompt ending where the verdict label goes
    build_request()     one vLLM request for (sample, response order, few-shot set)

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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from open_cascade.registry import JudgeConfig

LABELS = ("A", "B")
ASSISTANT_PREFIX = "[["

# label -> preference value, for each of the two response orderings
ORDERING_CONVERTERS = ({"A": 1, "B": 2}, {"B": 1, "A": 2})

# One scored simulation: {"A": p, "B": p}, or None if the judge answered off-format.
LabelProbs = Optional[Dict[str, float]]


def response_orderings(sample: Dict) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """(A, B) in the original order, then swapped. Index = ORDERING_CONVERTERS index."""
    first, second = sample["outputs"][0], sample["outputs"][1]
    return (first, second), (second, first)


def normalise_label_logprobs(step_logprobs: Dict, label_token_ids: Dict[str, int]) -> LabelProbs:
    """P(A), P(B) from one decode step's top logprobs, renormalised over the two labels.

    `step_logprobs` is vLLM's `{token_id: Logprob}` for the step. Returns None if neither
    label made the top-k, i.e. the judge is answering off-format.
    """
    logprobs = {
        label: step_logprobs[token_id].logprob
        for label, token_id in label_token_ids.items()
        if token_id in step_logprobs
    }

    if not logprobs:
        return None
    if len(logprobs) == 1:
        # floor the absent label just below the least likely returned token
        floor = min(lp.logprob for lp in step_logprobs.values()) - 1.0
        missing = next(label for label in LABELS if label not in logprobs)
        logprobs[missing] = floor

    offset = max(logprobs.values())
    unnormalised = {label: math.exp(lp - offset) for label, lp in logprobs.items()}
    total = sum(unnormalised.values())
    return {label: value / total for label, value in unnormalised.items()}


def average_simulations(n_samples: int, provenance: Sequence[Tuple[int, int]],
                        scored: Sequence[LabelProbs]) -> List[List[float]]:
    """Map each simulation back into preference-label space, then average per sample.

    `provenance[i] = (sample index, ordering index)` describes `scored[i]`.
    Returns `[P(preference=1), P(preference=2)]` per sample, or `[]` if every simulation
    for that sample failed to produce a usable label.
    """
    per_sample: List[List[Dict[int, float]]] = [[] for _ in range(n_samples)]
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
    return results


def simulation_details(n_samples: int, provenance: Sequence[Tuple[int, int, int]],
                       scored: Sequence[LabelProbs]) -> List[Dict]:
    """Keep each usable simulation after mapping it into original preference space.

    ``provenance`` entries are ``(sample index, annotator index, ordering index)``.
    These records make judge-setup ablations possible without another GPU scoring run.
    """
    details = [{"simulations": []} for _ in range(n_samples)]
    for (sample_idx, annotator, ordering), probs in zip(provenance, scored):
        if probs is None:
            continue
        converter = ORDERING_CONVERTERS[ordering]
        mapped = {converter[label]: probability for label, probability in probs.items()}
        details[sample_idx]["simulations"].append({
            "annotator": annotator,
            "ordering": ordering,
            "probs": [float(mapped[1]), float(mapped[2])],
        })
    return details


class BaseJudge(ABC):
    """An open-weight judge scored via teacher-forced A/B label logprobs.

    Args:
        model_name: key in `JUDGE_REGISTRY`.
        config: that judge's `JudgeConfig`.
        enable_prefix_caching: reuse the KV cache of the shared few-shot prefix. There are
            only N distinct prefixes in a run, so this is a large saving.
        max_fewshot_examples: use at most this many of each annotator's K examples.
            None = use all of them.
    """

    def __init__(self, model_name: str, config: JudgeConfig, enable_prefix_caching: bool = True,
                 max_fewshot_examples: Optional[int] = None):
        if max_fewshot_examples is not None and max_fewshot_examples < 0:
            raise ValueError("max_fewshot_examples must be non-negative")

        logging.getLogger("vllm").setLevel(logging.WARNING)
        from vllm import LLM  # imported here so the module is importable without a GPU

        self.model_name = model_name
        self.config = config
        self.max_fewshot_examples = max_fewshot_examples

        self.llm = LLM(**self._llm_kwargs(enable_prefix_caching))
        self.tokenizer = self._load_tokenizer()
        self.label_token_ids = self._resolve_label_token_ids()

    # subclass hooks
    @abstractmethod
    def _load_tokenizer(self):
        """Return the tokenizer used to encode prompts."""

    @abstractmethod
    def _probe_prompt(self) -> str:
        """A fully rendered prompt that ends exactly where the verdict label is written."""

    @abstractmethod
    def build_request(self, sample: Dict, assistant_a: str, assistant_b: str,
                      fewshot_examples: List[Dict]) -> Dict:
        """One vLLM request (a dict accepted by `LLM.generate`) for one simulation."""

    def _extra_llm_kwargs(self) -> Dict:
        """Modality-specific `vllm.LLM` arguments."""
        return {}

    # setup
    def _llm_kwargs(self, enable_prefix_caching: bool) -> Dict:
        llm_kwargs = dict(
            model=self.config.hf_name,
            seed=42,
            dtype=self.config.dtype,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            **self._extra_llm_kwargs(),
            **self.config.extra_llm_kwargs,
        )
        if self.config.quantization is not None:
            llm_kwargs["quantization"] = self.config.quantization
        if self.config.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.config.max_model_len
        return llm_kwargs

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
        probe = self._probe_prompt()
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

    # scoring
    def _generate(self, requests: List[Dict]):
        from vllm import SamplingParams

        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=1, logprobs=20)
        return self.llm.generate(requests, sampling_params, use_tqdm=False)

    def score_requests(self, requests: List[Dict]) -> List[LabelProbs]:
        """P(A), P(B) per request, renormalised over the two labels. None if neither ranked."""
        return [
            normalise_label_logprobs(output.outputs[0].logprobs[0], self.label_token_ids)
            for output in self._generate(requests)
        ]

    def simulate_annotators(self, sample: Dict,
                            fewshot_examples_list: List[List[Dict]]) -> List[float]:
        """Confidence for a single sample. Same signature as the authors' implementation."""
        return self.simulate_annotators_batch([sample], fewshot_examples_list)[0]

    def simulate_annotators_batch(self, samples: List[Dict],
                                  fewshot_examples_list: List[List[Dict]]) -> List[List[float]]:
        """Score many samples in one vLLM call.

        Requests are emitted grouped by annotator so consecutive prompts share a few-shot
        prefix, which is what makes prefix caching effective.

        Returns `[P(preference=1), P(preference=2)]` per sample, or `[]` if every simulation
        for that sample failed to produce a usable label.
        """
        return self.simulate_annotators_batch_with_details(
            samples, fewshot_examples_list)[0]

    def simulate_annotators_batch_with_details(
            self, samples: List[Dict], fewshot_examples_list: List[List[Dict]]
    ) -> Tuple[List[List[float]], List[Dict]]:
        """Return aggregate probabilities and the mapped per-run probabilities."""
        if self.max_fewshot_examples is not None:
            fewshot_examples_list = [
                examples[:self.max_fewshot_examples] for examples in fewshot_examples_list
            ]

        requests: List[Dict] = []
        provenance: List[Tuple[int, int, int]] = []

        for annotator, fewshot_examples in enumerate(fewshot_examples_list):
            for sample_idx, sample in enumerate(samples):
                for ordering, (a, b) in enumerate(response_orderings(sample)):
                    requests.append(self.build_request(sample, a, b, fewshot_examples))
                    provenance.append((sample_idx, annotator, ordering))

        scored = self.score_requests(requests)
        aggregate_provenance = [(sample_idx, ordering)
                                for sample_idx, _, ordering in provenance]
        results = average_simulations(len(samples), aggregate_provenance, scored)
        details = simulation_details(len(samples), provenance, scored)
        assert len(results) == len(samples)
        return results, details
