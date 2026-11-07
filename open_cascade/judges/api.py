"""Hosted judges behind an OpenAI-compatible chat API (OpenRouter, Gemini).

Same Simulated Annotators procedure as `BaseJudge`: for every sample, each of the N few-shot
sets is run in both response orderings, and the 2N verdict distributions are averaged with
`average_simulations`. The output is `[P(outputs[0]), P(outputs[1])]`, the same as for the
vLLM judges, so `score` and `evaluate` need no changes.

What differs is how one verdict distribution is obtained, because an API cannot be
teacher-forced on "[[" the way a local model can:

  confidence="logprobs"  The model answers "[[A]]" or "[[B]]" and the response carries
                         token logprobs. P(A), P(B) are read from the top alternatives at
                         the position where the letter was written, then renormalised
                         over the two. Closest to the local judges.
  confidence="votes"     The model's verdict is parsed from the text and counted as a vote,
                         `n_votes` times per simulation. With n_votes=1 at temperature 0
                         each simulation gives a hard 0/1 verdict, and graded confidence
                         comes only from averaging over the 2N simulations. For providers
                         without logprobs (Gemini).

Images are downscaled to `max_image_side` and sent inline as base64 JPEG.

API keys are read from the environment, never from configs:
    OPENROUTER_API_KEY   provider "openrouter"
    GEMINI_API_KEY       provider "gemini" (GOOGLE_API_KEY also accepted)
"""
import base64
import io
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from open_cascade.data import preferred_index
from open_cascade.judges.base import LABELS, LabelProbs, average_simulations, response_orderings
from open_cascade.registry import JudgeConfig


@dataclass(frozen=True)
class Provider:
    base_url: str
    key_env: Tuple[str, ...]
    headers: Tuple[Tuple[str, str], ...] = ()
    body_defaults: Tuple[Tuple[str, object], ...] = ()  # added to every request body


PROVIDERS: Dict[str, Provider] = {
    "openrouter": Provider(
        base_url="https://openrouter.ai/api/v1",
        key_env=("OPENROUTER_API_KEY",),
        headers=(("X-Title", "cascaded-selective-evaluation"),),
        body_defaults=(("usage", {"include": True}),),  # report $ cost per request
    ),
    "gemini": Provider(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
}

DEFAULT_API_OPTIONS = {
    "confidence": "logprobs",    # "logprobs" or "votes"
    "max_tokens": 16,
    "temperature": 0.0,          # votes mode with n_votes > 1 needs > 0 to be meaningful
    "n_votes": 1,                # votes mode: requests per simulation
    "top_logprobs": 5,           # some providers reject more than 5
    "concurrency": 8,            # requests in flight
    "max_image_side": 1024,      # px; longer side is downscaled to this
    "timeout": 120.0,            # seconds per request
    "max_retries": 6,            # on 429 / 5xx / network errors, exponential backoff
    "extra_body": {},            # merged into every request body (reasoning, routing, ...)
}

ANSWER_INSTRUCTION = '\n\nAnswer with exactly "[[A]]" or "[[B]]" and nothing else.'
VERDICT_RE = re.compile(r"\[\[\s*([AB])\s*\]\]")
LETTER_AFTER_BRACKETS_RE = re.compile(r"\[\[\s*([AB])")

# Failing this many requests in a row, with none succeeding, aborts the run: something is
# misconfigured (no logprobs returned, model off-format) and continuing just burns credit.
FAIL_FAST_AFTER = 20


class ApiAuthError(RuntimeError):
    pass


# ---------------------------------------------------------------- response parsing
def parse_verdict(text: Optional[str]) -> Optional[str]:
    """The last "[[A]]"/"[[B]]" in the text (reasoning models may mention both)."""
    if not text:
        return None
    matches = VERDICT_RE.findall(text)
    return matches[-1] if matches else None


def label_probs_from_logprobs(content_logprobs: List[Dict]) -> LabelProbs:
    """P(A), P(B) at the position where the verdict letter was generated.

    `content_logprobs` is OpenAI's `choices[0].logprobs.content`: one entry per generated
    token with `token`, `logprob` and `top_logprobs`. Tokenizers split "[[A]]" differently
    ("[[" "A" "]]", "[[A" "]]", ...), so the letter is located by character offset, and each
    alternative token at that position counts toward a label if it reads as the same prefix
    followed by that letter. Returns None if no verdict letter is found.
    """
    if not content_logprobs:
        return None

    starts, text = [], ""
    for entry in content_logprobs:
        starts.append(len(text))
        text += entry.get("token") or ""

    match = None
    for match in LETTER_AFTER_BRACKETS_RE.finditer(text):
        pass  # keep the last one, consistent with parse_verdict
    if match is None:
        return None

    letter_pos = match.start(1)
    idx = max(i for i, start in enumerate(starts) if start <= letter_pos)
    entry = content_logprobs[idx]
    token_prefix = text[starts[idx]:letter_pos]  # whatever precedes the letter in this token

    candidates = {entry["token"]: entry["logprob"]}
    for alt in entry.get("top_logprobs") or []:
        candidates.setdefault(alt["token"], alt["logprob"])

    per_label: Dict[str, List[float]] = {label: [] for label in LABELS}
    for token, logprob in candidates.items():
        if logprob is None or not token.startswith(token_prefix):
            continue
        rest = token[len(token_prefix):].lstrip()
        if rest[:1] in per_label:
            per_label[rest[:1]].append(logprob)

    logprobs = {label: _logsumexp(values) for label, values in per_label.items() if values}
    if not logprobs:
        return None
    if len(logprobs) == 1:
        # the other label is not in the top-k: floor it below the least likely alternative
        floor = min(lp for lp in candidates.values() if lp is not None) - 1.0
        missing = next(label for label in LABELS if label not in logprobs)
        logprobs[missing] = min(floor, next(iter(logprobs.values())) - 1.0)

    offset = max(logprobs.values())
    unnormalised = {label: math.exp(lp - offset) for label, lp in logprobs.items()}
    total = sum(unnormalised.values())
    return {label: value / total for label, value in unnormalised.items()}


def _logsumexp(values: List[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


# --------------------------------------------------------------------- images
@lru_cache(maxsize=1024)
def encode_image(path: str, max_side: int) -> str:
    """`data:` URL of the image, downscaled so its longer side is at most `max_side`."""
    from PIL import Image

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing image {image_path}. Run from the repo root (image_path is "
            "repo-relative), and build the media with `python get_data.py`.")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------- judge
class ApiJudge:
    """A judge served by a hosted API. Drop-in for `BaseJudge` in `scoring.score_judge`.

    Args:
        model_name: key in `JUDGE_REGISTRY`.
        config: that judge's `JudgeConfig`, with `provider` set.
        max_fewshot_examples: use at most this many of each annotator's K examples.
        api_key: overrides the provider's environment variable (tests, scripts).
        client: an `httpx.Client` to use instead of a new one (tests).
    """

    def __init__(self, model_name: str, config: JudgeConfig,
                 max_fewshot_examples: Optional[int] = None,
                 api_key: Optional[str] = None, client: Optional[httpx.Client] = None,
                 **_ignored):  # e.g. enable_prefix_caching, meaningless for an API
        if config.provider not in PROVIDERS:
            raise ValueError(f"Judge '{model_name}': unknown provider '{config.provider}'. "
                             f"Known: {sorted(PROVIDERS)}")
        unknown = sorted(set(config.api) - set(DEFAULT_API_OPTIONS))
        if unknown:
            raise ValueError(f"Judge '{model_name}': unknown api option(s) {unknown}. "
                             f"Allowed: {sorted(DEFAULT_API_OPTIONS)}")

        self.model_name = model_name
        self.config = config
        self.provider = PROVIDERS[config.provider]
        self.options = {**DEFAULT_API_OPTIONS, **config.api}
        self.max_fewshot_examples = max_fewshot_examples
        if self.options["confidence"] not in ("logprobs", "votes"):
            raise ValueError(f"api.confidence must be 'logprobs' or 'votes', "
                             f"got {self.options['confidence']!r}")

        if config.modality == "vlm":
            from model import vlm_prompts as prompts
        else:
            from model import prompts
        self.prompts = prompts

        api_key = api_key or self._api_key_from_env()
        headers = {"Authorization": f"Bearer {api_key}", **dict(self.provider.headers)}
        self.client = client or httpx.Client(base_url=self.provider.base_url, headers=headers,
                                             timeout=self.options["timeout"])

        self._lock = threading.Lock()
        self.stats = {"requests": 0, "ok": 0, "failed": 0, "retries": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        self.failure_reasons: Dict[str, int] = {}
        self.failure_examples: List[str] = []
        self._consecutive_failures = 0

    # what scoring.py prints after loading a judge
    @property
    def label_token_ids(self) -> str:
        return (f"n/a (API: {self.config.provider}/{self.config.hf_name}, "
                f"confidence={self.options['confidence']})")

    def _api_key_from_env(self) -> str:
        for name in self.provider.key_env:
            if os.environ.get(name):
                return os.environ[name]
        raise ApiAuthError(f"Set {' or '.join(self.provider.key_env)} to use judge "
                           f"'{self.model_name}' ({self.config.provider}).")

    # ------------------------------------------------------------ prompt building
    def _image_part(self, sample: Dict) -> Dict:
        url = encode_image(str(sample["image_path"]), int(self.options["max_image_side"]))
        return {"type": "image_url", "image_url": {"url": url}}

    def build_messages(self, sample: Dict, assistant_a: str, assistant_b: str,
                       fewshot_examples: List[Dict]) -> List[Dict]:
        """OpenAI-format messages for one simulation. Mirrors VLMJudge/TextJudge prompts."""
        p = self.prompts
        vlm = self.config.modality == "vlm"
        parts: List[Dict] = [{"type": "text", "text": p.fewshot_inst_prompt}]

        for example in fewshot_examples:
            if vlm:
                parts.append(self._image_part(example))
            parts.append({"type": "text", "text": p.fewshot_example_prompt.format(
                instruction=example["instruction"],
                assistant_a=example["outputs"][0],
                assistant_b=example["outputs"][1],
                preferred_response="[[A]]" if preferred_index(example) == 1 else "[[B]]",
            )})

        if vlm:
            parts.append(self._image_part(sample))
        parts.append({"type": "text", "text": p.fewshot_query_prompt.format(
            instruction=sample["instruction"], assistant_a=assistant_a,
            assistant_b=assistant_b) + ANSWER_INSTRUCTION})

        if not vlm:  # one string, joined exactly as TextJudge.format_user_prompt does
            parts = "\n".join(part["text"] for part in parts)
        return [{"role": "system", "content": p.system_prompt},
                {"role": "user", "content": parts}]

    def build_body(self, messages: List[Dict]) -> Dict:
        o = self.options
        body = {**dict(self.provider.body_defaults),
                "model": self.config.hf_name, "messages": messages,
                "max_tokens": o["max_tokens"], "temperature": o["temperature"]}
        if o["confidence"] == "logprobs":
            body["logprobs"] = True
            body["top_logprobs"] = o["top_logprobs"]
        return _deep_merge(body, o["extra_body"])

    # ------------------------------------------------------------------- HTTP
    def _post(self, body: Dict) -> Optional[Dict]:
        """POST /chat/completions with retries. None on a non-retryable request error."""
        delay = 2.0
        for attempt in range(self.options["max_retries"] + 1):
            try:
                response = self.client.post("/chat/completions", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as error:
                reason = f"network: {type(error).__name__}"
            else:
                if response.status_code in (401, 403):
                    raise ApiAuthError(f"{self.config.provider} rejected the API key "
                                       f"({response.status_code}): {response.text[:300]}")
                if response.status_code == 200:
                    data = response.json()
                    # OpenRouter reports upstream failures as 200 + {"error": ...}
                    if "error" not in data:
                        return data
                    reason = f"upstream error: {str(data['error'])[:200]}"
                elif response.status_code in (408, 429) or response.status_code >= 500:
                    reason = f"HTTP {response.status_code}"
                    retry_after = response.headers.get("retry-after")
                    if retry_after and retry_after.replace(".", "", 1).isdigit():
                        delay = max(delay, float(retry_after))
                else:
                    self._record_failure(f"HTTP {response.status_code}", response.text[:300])
                    return None

            if attempt == self.options["max_retries"]:
                self._record_failure("retries exhausted", reason)
                return None
            with self._lock:
                self.stats["retries"] += 1
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        return None

    def _record_failure(self, reason: str, detail: str = "") -> None:
        with self._lock:
            self.stats["failed"] += 1
            self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1
            if len(self.failure_examples) < 5:
                self.failure_examples.append(f"{reason}: {detail}"[:400])
            self._consecutive_failures += 1
            abort = self.stats["ok"] == 0 and self._consecutive_failures >= FAIL_FAST_AFTER
        if abort:
            raise RuntimeError(
                f"First {FAIL_FAST_AFTER} requests to '{self.model_name}' all failed. "
                f"Reasons: {self.failure_reasons}. Examples:\n  "
                + "\n  ".join(self.failure_examples)
                + "\nIf the reason is 'no logprobs', use the -votes variant of this judge.")

    def _record_usage(self, data: Dict) -> None:
        usage = data.get("usage") or {}
        with self._lock:
            self.stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
            self.stats["completion_tokens"] += usage.get("completion_tokens") or 0
            self.stats["cost_usd"] += float(usage.get("cost") or 0.0)

    # ---------------------------------------------------------------- scoring
    def score_messages(self, messages: List[Dict]) -> LabelProbs:
        """One simulation -> {"A": p, "B": p}, or None if the judge gave no usable verdict."""
        body = self.build_body(messages)
        n_votes = 1 if self.options["confidence"] == "logprobs" else self.options["n_votes"]

        votes = {label: 0 for label in LABELS}
        for _ in range(n_votes):
            with self._lock:
                self.stats["requests"] += 1
            data = self._post(body)
            if data is None:
                continue
            self._record_usage(data)
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")

            if self.options["confidence"] == "logprobs":
                logprobs = (choice.get("logprobs") or {}).get("content")
                if not logprobs:
                    self._record_failure("no logprobs in response",
                                         f"finish_reason={choice.get('finish_reason')} "
                                         f"content={content!r}")
                    return None
                probs = label_probs_from_logprobs(logprobs)
                if probs is None:
                    self._record_failure("no verdict in output",
                                         f"finish_reason={choice.get('finish_reason')} "
                                         f"content={content!r}")
                else:
                    self._record_ok()
                return probs

            verdict = parse_verdict(content)
            if verdict is None:
                self._record_failure("no verdict in output",
                                     f"finish_reason={choice.get('finish_reason')} "
                                     f"content={content!r}")
                continue
            self._record_ok()
            votes[verdict] += 1

        total = sum(votes.values())
        if total == 0:
            return None
        return {label: count / total for label, count in votes.items()}

    def _record_ok(self) -> None:
        with self._lock:
            self.stats["ok"] += 1
            self._consecutive_failures = 0

    def simulate_annotators(self, sample: Dict,
                            fewshot_examples_list: List[List[Dict]]) -> List[float]:
        return self.simulate_annotators_batch([sample], fewshot_examples_list)[0]

    def simulate_annotators_batch(self, samples: List[Dict],
                                  fewshot_examples_list: List[List[Dict]]) -> List[List[float]]:
        """`[P(preference=1), P(preference=2)]` per sample (`[]` if every simulation failed)."""
        if self.max_fewshot_examples is not None:
            fewshot_examples_list = [examples[:self.max_fewshot_examples]
                                     for examples in fewshot_examples_list]

        jobs, provenance = [], []
        for fewshot_examples in fewshot_examples_list:
            for sample_idx, sample in enumerate(samples):
                for ordering, (a, b) in enumerate(response_orderings(sample)):
                    jobs.append(self.build_messages(sample, a, b, fewshot_examples))
                    provenance.append((sample_idx, ordering))

        with ThreadPoolExecutor(max_workers=self.options["concurrency"]) as pool:
            scored = list(pool.map(self.score_messages, jobs))

        results = average_simulations(len(samples), provenance, scored)
        assert len(results) == len(samples)
        return results

    # -------------------------------------------------------------- reporting
    def usage_summary(self) -> str:
        s = self.stats
        lines = [f"API usage for {self.model_name}: {s['requests']} requests "
                 f"({s['ok']} ok, {s['failed']} failed, {s['retries']} retries), "
                 f"{s['prompt_tokens']:,} prompt + {s['completion_tokens']:,} completion tokens"
                 + (f", ${s['cost_usd']:.4f}" if s["cost_usd"] else "")]
        if self.failure_reasons:
            lines.append(f"  failures: {self.failure_reasons}")
            lines.extend(f"    e.g. {example}" for example in self.failure_examples[:3])
        return "\n".join(lines)

    def shutdown(self) -> None:
        print(self.usage_summary())
        self.client.close()


def _deep_merge(base: Dict, extra: Dict) -> Dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
