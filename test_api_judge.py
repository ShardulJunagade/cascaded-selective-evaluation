"""Smoke-test a hosted (API) judge end to end before spending real budget on `score`.

    python test_api_judge.py --judge kimi-k2.5 --n 3
    python test_api_judge.py --judge gemini-flash --n 3
    python test_api_judge.py --judge kimi-k2.5 --dry-run      # no network, no key needed

Steps:
  1. ping     one tiny text request: checks the key, the model id, and that the response
              is in the format the judge's confidence mode needs (logprobs present, verdict
              parseable).
  2. score    the first --n samples of --split through the real Simulated Annotators path
              (judge.simulate_annotators_batch, exactly what `score` calls), printing each
              judgement next to the human label.
  3. report   agreement, confidence, failures, tokens, cost, and the projected cost of
              scoring the config's splits.

Keys come from the environment: OPENROUTER_API_KEY, GEMINI_API_KEY.
"""
import json
import sys
import time
from argparse import ArgumentParser

from open_cascade.data import load_fewshot, preferred_index, read_jsonl
from open_cascade.judges import build_judge
from open_cascade.judges.api import ApiAuthError, label_probs_from_logprobs, parse_verdict
from open_cascade.judges.base import response_orderings
from open_cascade.registry import JUDGE_REGISTRY

EXPORT = "data/vlm_v2_export"


def shorten(value, limit=160):
    """Message payload for printing: base64 images and long texts truncated."""
    if isinstance(value, dict):
        return {k: shorten(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten(v, limit) for v in value]
    if isinstance(value, str):
        if value.startswith("data:image"):
            return f"<{value[:23]}... {len(value) // 1024} KB>"
        if len(value) > limit:
            return value[:limit] + f"... <{len(value)} chars>"
    return value


def ping(judge) -> bool:
    """One tiny request with the judge's settings. True if it yields a usable verdict."""
    return send_ping(judge, judge.build_body(PING_MESSAGES))


PING_MESSAGES = [{"role": "user", "content":
                  'Which letter comes first in the alphabet, A or B? '
                  'Answer with exactly "[[A]]" or "[[B]]" and nothing else.'}]

# --probe: ways to stop a reasoning model from thinking before the verdict, per provider.
# Each is merged into the judge's request body. Providers differ in which they honour.
_NO_FALLBACK = {"allow_fallbacks": False, "require_parameters": True}
PROBES = {
    "as configured": {},
    "novita + reasoning off": {"provider": {"order": ["novita"], **_NO_FALLBACK}},
    "novita + chat_template thinking=false": {
        "provider": {"order": ["novita"], **_NO_FALLBACK},
        "chat_template_kwargs": {"thinking": False}},
    "venice + chat_template thinking=false": {
        "provider": {"order": ["venice"], **_NO_FALLBACK},
        "chat_template_kwargs": {"thinking": False}},
    "novita + thinking.type=disabled": {
        "provider": {"order": ["novita"], **_NO_FALLBACK},
        "thinking": {"type": "disabled"}},
    "venice + thinking.type=disabled": {
        "provider": {"order": ["venice"], **_NO_FALLBACK},
        "thinking": {"type": "disabled"}},
    "let it reason, max_tokens=1024": {"max_tokens": 1024},
}


def send_ping(judge, body) -> bool:
    start = time.time()
    judge.stats["requests"] += 1
    data = judge._post(body)
    elapsed = time.time() - start
    if data is None:
        print(f"  FAILED: {judge.failure_examples[-1] if judge.failure_examples else '?'}")
        return False
    judge._record_usage(data)

    choice = data["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    usage = data.get("usage") or {}
    print(f"  served by: {data.get('provider', '?')}  ({elapsed:.1f}s, "
          f"{usage.get('completion_tokens', '?')} completion tokens)")
    print(f"  content: {shorten(content, 200)!r}  finish_reason: {choice.get('finish_reason')}")
    if message.get("reasoning"):
        print(f"  NOTE: separate reasoning returned ({len(message['reasoning'])} chars)")
    if choice.get("finish_reason") == "length" and parse_verdict(content) is None:
        print("  -> cut off before the verdict: the model is thinking first")

    if judge.options["confidence"] == "logprobs":
        logprobs = (choice.get("logprobs") or {}).get("content")
        if not logprobs:
            print("  FAILED: no logprobs returned (provider ignores the parameter). "
                  "Use the -votes variant of this judge.")
            return False
        if len(logprobs) <= 8:
            print(f"  tokens: {[entry['token'] for entry in logprobs]}")
        probs = label_probs_from_logprobs(logprobs)
        print(f"  P(A), P(B) at verdict: {probs}")
        return probs is not None

    verdict = parse_verdict(content)
    print(f"  parsed verdict: {verdict}")
    return verdict is not None


def probe(judge) -> None:
    from open_cascade.judges.api import _deep_merge

    results = {}
    for name, overrides in PROBES.items():
        print(f"\n--- {name}")
        body = _deep_merge(judge.build_body(PING_MESSAGES), overrides)
        try:
            results[name] = send_ping(judge, body)
        except Exception as error:  # keep probing the rest
            print(f"  ERROR: {error}")
            results[name] = False
    print("\nsummary:")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")


def main():
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--judge", default="kimi-k2.5",
                        choices=[n for n, c in JUDGE_REGISTRY.items() if c.provider])
    parser.add_argument("--split", default=f"{EXPORT}/calibration.jsonl")
    parser.add_argument("--fewshot", default=f"{EXPORT}/fewshot.maj.jsonl")
    parser.add_argument("--n", type=int, default=3, help="samples to score")
    parser.add_argument("--k", type=int, default=1, help="few-shot examples per annotator")
    parser.add_argument("--annotators", type=int, default=None,
                        help="use only the first N simulated annotators (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the request for one simulation and exit (no network)")
    parser.add_argument("--skip-ping", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="try several provider / thinking-off settings and report which "
                             "give a clean verdict (a few cents), then exit")
    parser.add_argument("--out", default=None, help="also write the judgements to this jsonl")
    args = parser.parse_args()

    samples = read_jsonl(args.split)[:args.n]
    fewshot = load_fewshot(args.fewshot)[:args.annotators]
    n_ann = len(fewshot)
    requests_per_sample = 2 * n_ann
    print(f"judge {args.judge}: {JUDGE_REGISTRY[args.judge].provider}/"
          f"{JUDGE_REGISTRY[args.judge].hf_name}")
    print(f"{len(samples)} samples from {args.split}; N={n_ann} annotators, K={args.k} "
          f"-> {requests_per_sample} requests per sample")

    try:
        judge = build_judge(args.judge, max_fewshot_examples=args.k,
                            **({"api_key": "dry-run"} if args.dry_run else {}))
    except ApiAuthError as error:
        sys.exit(f"ERROR: {error}")
    print(f"confidence mode: {judge.options['confidence']}")

    if args.dry_run:
        a, b = response_orderings(samples[0])[0]
        body = judge.build_body(judge.build_messages(samples[0], a, b, fewshot[0][:args.k]))
        print(json.dumps(shorten(body), indent=2, ensure_ascii=False))
        payload_kb = len(json.dumps(body)) / 1024
        print(f"\nrequest size: {payload_kb:.0f} KB. Dry run: nothing sent.")
        return

    if args.probe:
        try:
            probe(judge)
        finally:
            judge.shutdown()
        return

    baseline, elapsed, judged, agree = dict(judge.stats), 0.0, [], 0
    try:
        if not args.skip_ping:
            print("\n[1] ping")
            if not ping(judge):
                sys.exit("Ping failed; fix the above before scoring.")

        print(f"\n[2] scoring {len(samples)} samples")
        baseline = dict(judge.stats)  # so per-sample numbers exclude the ping
        judged, agree = [], 0
        start = time.time()
        for i, sample in enumerate(samples):
            t0 = time.time()
            probs = judge.simulate_annotators_batch([sample], fewshot)[0]
            human = preferred_index(sample)
            if not probs:
                print(f"  #{i}: every simulation failed ({time.time() - t0:.1f}s)")
                continue
            predicted = 1 if probs[0] >= probs[1] else 2
            agree += predicted == human
            judged.append({**sample, "probs": probs})
            src = (sample.get("source") or {}).get("dataset", "?")
            print(f"  #{i} [{src}] probs=[{probs[0]:.3f}, {probs[1]:.3f}] "
                  f"pred={predicted} human={human} {'OK ' if predicted == human else 'BAD'} "
                  f"conf={max(probs):.3f} ({time.time() - t0:.1f}s)")
        elapsed = time.time() - start
    except ApiAuthError as error:
        sys.exit(f"ERROR: {error}")
    finally:
        print("\n[3] report")
        judge.shutdown()

    if judged:
        print(f"agreement with human: {agree}/{len(judged)}  "
              f"mean confidence: {sum(max(s['probs']) for s in judged) / len(judged):.3f}  "
              f"time/sample: {elapsed / len(samples):.1f}s")
    cost = judge.stats["cost_usd"] - baseline["cost_usd"]
    if cost and samples:
        per_sample = cost / len(samples)
        print(f"cost/sample ~ ${per_sample:.4f} at N={n_ann}, K={args.k}  ->  "
              f"1,000 samples ~ ${per_sample * 1000:.2f}, "
              f"full eval_pool (25,308) ~ ${per_sample * 25308:,.0f}")
    elif samples:
        tokens = (judge.stats["prompt_tokens"] - baseline["prompt_tokens"]) / len(samples)
        print(f"~{tokens:,.0f} prompt tokens/sample (provider reported no $ cost; "
              f"multiply by its price)")

    if args.out and judged:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(json.dumps(s) for s in judged) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
