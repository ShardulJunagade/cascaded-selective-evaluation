"""Hosted (API) judges: response parsing, Simulated Annotators averaging, retries, and the
`score` stage end to end. All HTTP is mocked; no network or API key needed."""
import json
import math

import httpx
import pytest
from PIL import Image

import open_cascade.judges as judges_pkg
import open_cascade.judges.api as api
from open_cascade.config import config_from_dict
from open_cascade.data import read_jsonl, write_jsonl
from open_cascade.judges.api import ApiAuthError, ApiJudge, label_probs_from_logprobs, parse_verdict
from open_cascade.registry import JudgeConfig
from open_cascade.scoring import score_judge


# ------------------------------------------------------------------ helpers
def lp_entry(token, logprob, alts=()):
    return {"token": token, "logprob": logprob,
            "top_logprobs": [{"token": t, "logprob": l} for t, l in alts]}


def completion(content, logprobs=None, cost=0.001):
    choice = {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"choices": [choice], "usage": {"prompt_tokens": 100, "completion_tokens": 3,
                                           "cost": cost}}


def query_text(body):
    """The query (last) text part of the user message."""
    content = body["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return [part["text"] for part in content if part["type"] == "text"][-1]


def prefers_good(body):
    """A judge that likes the response containing 'GOOD', with P = 0.9."""
    query = query_text(body).split("[Instruction]")[-1]
    a_part = query.split("[Assistant A's response]")[1].split("[Assistant B's response]")[0]
    return "A" if "GOOD" in a_part else "B"


def make_judge(handler, modality="text", **api_options):
    config = JudgeConfig(hf_name="test/model", modality=modality, provider="openrouter",
                         api=api_options)
    client = httpx.Client(base_url="https://test", transport=httpx.MockTransport(handler))
    return ApiJudge("test-judge", config, api_key="k", client=client)


def logprob_handler(request):
    body = json.loads(request.content)
    winner = prefers_good(body)
    loser = "B" if winner == "A" else "A"
    entries = [lp_entry("[[", -0.01),
               lp_entry(winner, math.log(0.9), [(winner, math.log(0.9)), (loser, math.log(0.1))]),
               lp_entry("]]", -0.01)]
    return httpx.Response(200, json=completion(f"[[{winner}]]", entries))


SAMPLE = {"instruction": "q", "outputs": ["GOOD answer", "bad answer"],
          "preferences": {"human": 1}}
SAMPLE_SWAPPED = {"instruction": "q2", "outputs": ["bad answer", "GOOD answer"],
                  "preferences": {"human": 2}}
FEWSHOT = [[{"instruction": "ex", "outputs": ["x", "y"], "preferences": {"annotator_7": 2}}]] * 2


# ------------------------------------------------------------------ parsing
def test_parse_verdict_takes_last():
    assert parse_verdict("I considered [[A]] but [[B]]") == "B"
    assert parse_verdict("no verdict") is None
    assert parse_verdict(None) is None


def test_logprobs_split_tokens():
    entries = [lp_entry("[[", -0.01), lp_entry("B", math.log(0.7),
               [("B", math.log(0.7)), ("A", math.log(0.2)), ("C", math.log(0.05))]),
               lp_entry("]]", 0.0)]
    probs = label_probs_from_logprobs(entries)
    assert probs["B"] == pytest.approx(0.7 / 0.9)
    assert probs["A"] == pytest.approx(0.2 / 0.9)


def test_logprobs_merged_token():
    # tokenizer emits "[[A" as one token; alternatives share the "[[" prefix
    entries = [lp_entry("[[A", math.log(0.6), [("[[A", math.log(0.6)), ("[[B", math.log(0.3)),
                                                ("[[", math.log(0.05))]),
               lp_entry("]]", 0.0)]
    probs = label_probs_from_logprobs(entries)
    assert probs["A"] == pytest.approx(0.6 / 0.9)


def test_logprobs_merges_variants_of_same_label():
    entries = [lp_entry("[[", 0.0), lp_entry("A", math.log(0.5),
               [("A", math.log(0.5)), (" A", math.log(0.2)), ("B", math.log(0.3))])]
    assert label_probs_from_logprobs(entries)["A"] == pytest.approx(0.7)


def test_logprobs_floors_missing_label():
    entries = [lp_entry("[[", 0.0), lp_entry("A", -0.01, [("A", -0.01), ("The", -6.0)])]
    probs = label_probs_from_logprobs(entries)
    assert probs["A"] > 0.99 and 0 < probs["B"] < 0.01


def test_logprobs_no_verdict():
    assert label_probs_from_logprobs([lp_entry("Hello", -0.1)]) is None
    assert label_probs_from_logprobs([]) is None


# ------------------------------------------------------------------ judge
def test_logprobs_mode_maps_swapped_order_back():
    judge = make_judge(logprob_handler)
    probs = judge.simulate_annotators_batch([SAMPLE, SAMPLE_SWAPPED], FEWSHOT)
    assert probs[0] == pytest.approx([0.9, 0.1])   # GOOD is outputs[0]
    assert probs[1] == pytest.approx([0.1, 0.9])   # GOOD is outputs[1]
    assert judge.stats["requests"] == 2 * 2 * 2     # samples x annotators x orderings
    assert judge.stats["cost_usd"] == pytest.approx(0.008)


def test_request_body_and_prompt():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return logprob_handler(request)

    judge = make_judge(handler, extra_body={"provider": {"require_parameters": True}})
    judge.simulate_annotators_batch([SAMPLE], FEWSHOT[:1])
    body = seen[0]
    assert body["model"] == "test/model" and body["logprobs"] is True
    assert body["provider"] == {"require_parameters": True}
    assert body["usage"] == {"include": True}
    user = body["messages"][1]["content"]
    assert "[Preferred response]\n[[B]]" in user          # few-shot label via preferred_index
    assert user.rstrip().endswith('"[[A]]" or "[[B]]" and nothing else.')


def test_votes_mode_hard_verdicts():
    def handler(request):
        return httpx.Response(200, json=completion(f"[[{prefers_good(json.loads(request.content))}]]"))

    judge = make_judge(handler, confidence="votes")
    assert judge.simulate_annotators_batch([SAMPLE], FEWSHOT) == [pytest.approx([1.0, 0.0])]


def test_votes_mode_drops_off_format_answers():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=completion("[[A]]" if calls["n"] % 2 else "dunno"))

    judge = make_judge(handler, confidence="votes", concurrency=1)
    probs = judge.simulate_annotators_batch([SAMPLE], FEWSHOT)[0]
    assert len(probs) == 2 and judge.failure_reasons == {"no verdict in output": 2}


def test_retries_on_429(monkeypatch):
    monkeypatch.setattr(api.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        return logprob_handler(request)

    judge = make_judge(handler, concurrency=1)
    assert judge.simulate_annotators_batch([SAMPLE], FEWSHOT[:1])[0] == pytest.approx([0.9, 0.1])
    assert judge.stats["retries"] == 1


def test_bad_key_raises():
    judge = make_judge(lambda r: httpx.Response(401, text="invalid key"))
    with pytest.raises(ApiAuthError):
        judge.simulate_annotators_batch([SAMPLE], FEWSHOT)


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ApiAuthError, match="OPENROUTER_API_KEY"):
        ApiJudge("x", JudgeConfig(hf_name="m", provider="openrouter"))


def test_fails_fast_when_no_logprobs():
    judge = make_judge(lambda r: httpx.Response(200, json=completion("[[A]]")))
    many = [dict(SAMPLE, instruction=f"q{i}") for i in range(api.FAIL_FAST_AFTER)]
    with pytest.raises(RuntimeError, match="no logprobs"):
        judge.simulate_annotators_batch(many, FEWSHOT)


def test_unknown_api_option_rejected():
    with pytest.raises(ValueError, match="unknown api option"):
        make_judge(logprob_handler, confidnce="votes")


def test_vlm_messages_carry_images(tmp_path):
    image = tmp_path / "img.png"
    Image.new("RGB", (3000, 1500), "red").save(image)
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return logprob_handler(request)

    judge = make_judge(handler, modality="vlm", max_image_side=512)
    sample = dict(SAMPLE, image_path=str(image))
    fewshot = [[dict(FEWSHOT[0][0], image_path=str(image))]]
    judge.simulate_annotators_batch([sample], fewshot)

    parts = seen[0]["messages"][1]["content"]
    images = [p for p in parts if p["type"] == "image_url"]
    assert len(images) == 2   # one few-shot example + the query
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


# ------------------------------------------------------------- score stage
def test_score_stage_with_api_judge(tmp_path, monkeypatch):
    data_dir = tmp_path / "export"
    samples = [dict(SAMPLE, instruction=f"q{i}") for i in range(5)]
    write_jsonl(samples, data_dir / "calibration.jsonl")
    write_jsonl([{"annotator": "a", "evaluated_samples": FEWSHOT[0]}], data_dir / "fewshot.maj.jsonl")

    config = config_from_dict({
        "name": "t", "judges": ["test-judge"], "result_dir": str(tmp_path / "result"),
        "data": {"builder": "prebuilt", "out_dir": str(data_dir), "split_subdir": "",
                 "fewshot_name": "fewshot.maj.jsonl"},
        "scoring": {"splits": ["calibration"], "max_samples": 3, "chunk_size": 2},
    })
    monkeypatch.setattr("open_cascade.scoring.resolve_judge",
                        lambda name: JudgeConfig(hf_name="test/model", provider="openrouter"))
    monkeypatch.setattr(judges_pkg, "build_judge",
                        lambda name, **kwargs: make_judge(logprob_handler))

    score_judge(config, "test-judge")
    out = read_jsonl(tmp_path / "result" / "test-judge.calibration.jsonl")
    assert [s["instruction"] for s in out] == ["q0", "q1", "q2"]
    assert all(s["probs"] == pytest.approx([0.9, 0.1]) for s in out)
