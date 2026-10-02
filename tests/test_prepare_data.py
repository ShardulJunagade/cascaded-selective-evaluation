"""Tests for the dataset pipeline. No network, no GPU.

    pip install pytest && pytest tests/test_prepare_data.py -q

The three things worth guarding are the three things that were wrong or absent
before: the label must follow its response through a position swap, splits must
not leak an image across a boundary, and a given seed must reproduce a given
file.
"""

from __future__ import annotations

import pytest

from prepare_data import stages
from prepare_data.export import to_instance
from prepare_data.funnel import Funnel
from prepare_data.schema import (Annotation, Record, ValidationError,
                                 aggregate_annotations, make_group_id, make_uid,
                                 validate)


def make_record(uid: str, *, source: str = "test", label: str = "A",
                response_a: str = "response one", response_b: str = "response two",
                phash: str = "ph0", instruction: str = "describe the image",
                annotations=None, is_tie: bool = False,
                real_annotator_ids: bool = True) -> Record:
    annotations = annotations or []
    return Record(
        uid=uid,
        source=source,
        group_id=make_group_id(source, [phash], instruction),
        modality="image",
        media=[f"media/xx/{phash}.jpg"],
        media_hash=[phash],
        instruction=instruction,
        response_a=response_a,
        response_b=response_b,
        model_a="model_one",
        model_b="model_two",
        label=None if is_tie else label,
        label_origin="majority",
        annotations=[a.to_dict() for a in annotations],
        n_ann=len(annotations),
        agreement=None,
        is_tie=is_tie,
        meta={"len_a": len(response_a), "len_b": len(response_b),
              "real_annotator_ids": real_annotator_ids},
    )


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def test_validate_accepts_a_well_formed_record():
    validate(make_record("u1"))


@pytest.mark.parametrize("mutate, fragment", [
    (lambda r: setattr(r, "label", "C"), "label"),
    (lambda r: setattr(r, "response_b", "   "), "empty response"),
    (lambda r: setattr(r, "modality", "text"), "modality"),
    (lambda r: setattr(r, "media", []), "no media"),
    (lambda r: setattr(r, "n_ann", 7), "n_ann"),
    (lambda r: setattr(r, "agreement", 1.5), "agreement"),
])
def test_validate_rejects_malformed_records(mutate, fragment):
    record = make_record("u1")
    mutate(record)
    with pytest.raises(ValidationError) as excinfo:
        validate(record)
    assert fragment in str(excinfo.value)


def test_tie_record_must_not_carry_a_label():
    record = make_record("u1", is_tie=True)
    record.label = "A"
    with pytest.raises(ValidationError):
        validate(record)


def test_aggregate_annotations_majority_and_agreement():
    summary = aggregate_annotations([
        Annotation("w1", "A"), Annotation("w2", "A"), Annotation("w3", "B")])
    assert summary == {"label": "A", "agreement": pytest.approx(2 / 3),
                       "is_tie": False, "n_ann": 3}


def test_aggregate_annotations_even_split_is_a_tie():
    summary = aggregate_annotations([Annotation("w1", "A"), Annotation("w2", "B")])
    assert summary["is_tie"] and summary["label"] is None


def test_uid_is_stable_across_calls():
    assert make_uid("src", 1, "x") == make_uid("src", 1, "x")
    assert make_uid("src", 1, "x") != make_uid("src", 2, "x")


# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

def test_hygiene_drops_identical_empty_and_double_refusals():
    records = [
        make_record("keep"),
        make_record("identical", response_a="Same Text", response_b="same text"),
        make_record("empty", response_b="   "),
        make_record("refusals", response_a="I'm sorry, I can't help with that.",
                    response_b="I cannot assist with this request."),
    ]
    kept = stages.hygiene(records, Funnel())
    assert [r.uid for r in kept] == ["keep"]


def test_hygiene_flags_but_keeps_oversized_pairs():
    huge = make_record("huge", response_a="x" * 400_000, response_b="y" * 400_000)
    kept = stages.hygiene([huge], Funnel())
    assert len(kept) == 1
    assert kept[0].meta["over_context_budget"] is True


# --------------------------------------------------------------------------
# dedup
# --------------------------------------------------------------------------

def test_dedup_keeps_the_higher_priority_source():
    shared = {"phash": "same", "instruction": "what is in the picture"}
    records = [
        make_record("low", source="rlhf_v", **shared),
        make_record("high", source="visit_bench", **shared),
    ]
    kept = stages.dedup(records, Funnel())
    assert [r.source for r in kept] == ["visit_bench"]


def test_dedup_is_order_independent():
    shared = {"phash": "same", "instruction": "what is in the picture"}
    a = make_record("low", source="rlhf_v", **shared)
    b = make_record("high", source="visit_bench", **shared)
    assert ([r.uid for r in stages.dedup([a, b], Funnel())]
            == [r.uid for r in stages.dedup([b, a], Funnel())])


def test_dedup_keeps_distinct_instructions_on_one_image():
    records = [
        make_record("one", phash="img", instruction="describe this"),
        make_record("two", phash="img", instruction="count the people"),
    ]
    assert len(stages.dedup(records, Funnel())) == 2


def test_dedup_keeps_several_model_pairs_on_one_prompt():
    # VisIT-Bench pits seven models against each other on the same image and
    # instruction. Those are distinct comparisons, not duplicates.
    records = [
        make_record("pair1", source="visit_bench", phash="img",
                    response_a="llava says x", response_b="owl says y"),
        make_record("pair2", source="visit_bench", phash="img",
                    response_a="blip says z", response_b="minigpt says w"),
        make_record("pair3", source="visit_bench", phash="img",
                    response_a="llava says x", response_b="blip says z"),
    ]
    assert len(stages.dedup(records, Funnel())) == 3


def test_dedup_drops_an_exact_repeat_within_one_source():
    records = [
        make_record("first", source="visit_bench", phash="img"),
        make_record("again", source="visit_bench", phash="img"),
    ]
    assert len(stages.dedup(records, Funnel())) == 1


def test_dedup_treats_a_swapped_pair_as_the_same_instance():
    records = [
        make_record("ab", source="visit_bench", phash="img",
                    response_a="alpha", response_b="beta"),
        make_record("ba", source="visit_bench", phash="img",
                    response_a="beta", response_b="alpha"),
    ]
    assert len(stages.dedup(records, Funnel())) == 1


def test_dedup_keeps_distinct_comparisons_across_sources():
    # Two benchmarks asking the same question of the same image still compare
    # different model pairs. Those are distinct comparisons, not repeats, and
    # the group-aware split already stops the shared image from straddling a
    # calibration/test boundary.
    shared = {"phash": "img", "instruction": "describe this"}
    records = [
        make_record("ja1", source="judge_anything", response_a="p", response_b="q",
                    **shared),
        make_record("ja2", source="judge_anything", response_a="r", response_b="s",
                    **shared),
        make_record("vb", source="visit_bench", response_a="t", response_b="u",
                    **shared),
    ]
    assert len(stages.dedup(records, Funnel())) == 3


def test_dedup_drops_the_same_comparison_seen_in_two_sources():
    shared = {"phash": "img", "instruction": "describe this",
              "response_a": "same one", "response_b": "same two"}
    records = [
        make_record("weak", source="rlhf_v", **shared),
        make_record("strong", source="visit_bench", **shared),
    ]
    kept = stages.dedup(records, Funnel())
    assert [r.source for r in kept] == ["visit_bench"]


# --------------------------------------------------------------------------
# tie routing
# --------------------------------------------------------------------------

def test_route_ties_separates_tied_instances():
    main, ties = stages.route_ties(
        [make_record("a"), make_record("t", is_tie=True)], Funnel())
    assert [r.uid for r in main] == ["a"]
    assert [r.uid for r in ties] == ["t"]


# --------------------------------------------------------------------------
# position randomization - the defect this pipeline exists to fix
# --------------------------------------------------------------------------

def test_swap_moves_the_label_with_the_response():
    records = [make_record(f"u{i}", label="A") for i in range(200)]
    for record in stages.randomize_positions(records, seed=42):
        # "A" must always name whichever slot holds the originally preferred text.
        preferred = record.response_a if record.label == "A" else record.response_b
        assert preferred == "response one"


def test_swap_also_moves_model_names_and_per_annotator_labels():
    records = [make_record(f"u{i}", label="A", annotations=[Annotation("w1", "A")])
               for i in range(50)]
    swapped = [r for r in stages.randomize_positions(records, seed=42) if r.swapped]
    assert swapped, "seed produced no swaps; pick another"

    for record in swapped:
        assert (record.model_a, record.model_b) == ("model_two", "model_one")
        assert record.annotations == [{"annotator": "w1", "label": "B"}]
        assert (record.meta["len_a"], record.meta["len_b"]) == (len("response two"),
                                                                len("response one"))


def test_randomization_breaks_the_constant_label():
    records = [make_record(f"u{i}", label="A") for i in range(500)]
    out = stages.randomize_positions(records, seed=42)
    share_a = sum(r.label == "A" for r in out) / len(out)
    assert 0.40 < share_a < 0.60, f"labels still skewed: {share_a:.2f} are A"


def test_randomization_is_deterministic_and_independent_of_order():
    first = {r.uid: r.swapped for r in
             stages.randomize_positions([make_record(f"u{i}") for i in range(50)], seed=7)}
    shuffled = [make_record(f"u{i}") for i in reversed(range(50))]
    second = {r.uid: r.swapped for r in stages.randomize_positions(shuffled, seed=7)}
    assert first == second


def test_randomization_changes_with_the_seed():
    a = [r.swapped for r in
         stages.randomize_positions([make_record(f"u{i}") for i in range(100)], seed=1)]
    b = [r.swapped for r in
         stages.randomize_positions([make_record(f"u{i}") for i in range(100)], seed=2)]
    assert a != b


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def _split_corpus(n: int = 300):
    # Two records share each image, under different instructions, so a split
    # that only respected group_id would leak.
    records = []
    for i in range(n):
        records.append(make_record(f"a{i}", phash=f"img{i}", instruction=f"q{i} first"))
        records.append(make_record(f"b{i}", phash=f"img{i}", instruction=f"q{i} second"))
    return records


def test_split_has_no_image_or_group_leakage():
    splits = stages.split_groups(_split_corpus(), seed=42)
    stages.assert_no_leakage(splits)          # raises on leakage
    assert set(splits) == {"fewshot", "dev", "eval_pool"}
    assert all(splits[name] for name in splits)


def test_split_respects_the_requested_fractions_approximately():
    records = _split_corpus(400)
    splits = stages.split_groups(records, seed=42)
    share = len(splits["eval_pool"]) / len(records)
    assert 0.7 < share < 0.9, f"eval_pool share {share:.2f} far from 0.8"


def test_split_is_reproducible_for_a_seed():
    first = {k: [r.uid for r in v] for k, v in
             stages.split_groups(_split_corpus(), seed=42).items()}
    second = {k: [r.uid for r in v] for k, v in
              stages.split_groups(_split_corpus(), seed=42).items()}
    assert first == second


def test_fewshot_split_concentrates_prolific_annotators():
    # A purely random split scatters an annotator's votes across all three
    # splits, so nobody clears K+1 inside fewshot and the Ind. pools come out
    # empty. The split must gather them instead.
    records = []
    for i in range(30):                      # one prolific annotator
        records.append(make_record(f"p{i}", phash=f"pimg{i}", instruction=f"pq{i}",
                                   annotations=[Annotation("prolific", "A")]))
    for i in range(270):                     # no annotator signal
        records.append(make_record(f"n{i}", phash=f"nimg{i}", instruction=f"nq{i}",
                                   real_annotator_ids=False))

    splits = stages.split_groups(records, seed=42, k=2)
    stages.assert_no_leakage(splits)

    pools = stages.individual_pools(splits["fewshot"], k=2, n=3)
    assert [p["annotator"] for p in pools] == ["prolific"]

    in_fewshot = sum(1 for r in splits["fewshot"] if r.uid.startswith("p"))
    assert in_fewshot >= 3, f"only {in_fewshot} prolific records reached fewshot"


def test_fewshot_concentration_does_not_swallow_the_split():
    records = [make_record(f"p{i}", phash=f"img{i}", instruction=f"q{i}",
                           annotations=[Annotation(f"ann{i % 3}", "A")])
               for i in range(300)]
    splits = stages.split_groups(records, seed=42, k=2)
    share = len(splits["fewshot"]) / len(records)
    assert share <= 0.15, f"fewshot took {share:.0%} of the corpus"
    assert len(splits["eval_pool"]) > len(splits["fewshot"])


def test_assert_no_leakage_catches_a_shared_image():
    shared = make_record("x", phash="img")
    other = make_record("y", phash="img", instruction="different question")
    with pytest.raises(AssertionError):
        stages.assert_no_leakage({"dev": [shared], "eval_pool": [other]})


# --------------------------------------------------------------------------
# few-shot pools
# --------------------------------------------------------------------------

def test_individual_pools_skip_annotators_below_k_plus_one():
    records = [make_record(f"u{i}", annotations=[Annotation("prolific", "A")])
               for i in range(5)]
    records += [make_record("v", annotations=[Annotation("one_shot", "A")])]

    pools = stages.individual_pools(records, k=2, n=5)
    assert [p["annotator"] for p in pools] == ["prolific"]
    assert len(pools[0]["records"]) == 2


def test_majority_pools_are_disjoint():
    records = [make_record(f"u{i}") for i in range(20)]
    pools = stages.majority_pools(records, k=2, n=5)
    uids = [r.uid for pool in pools for r in pool["records"]]
    assert len(uids) == len(set(uids)) == 10


def test_majority_pools_are_stratified_across_sources():
    # A uniform draw over a corpus dominated by one source would demonstrate
    # only that source's style to a judge that has to grade all of them.
    records = [make_record(f"big{i}", source="visionarena", phash=f"b{i}")
               for i in range(200)]
    records += [make_record(f"small{i}", source="rlhf_v", phash=f"s{i}")
                for i in range(10)]

    pools = stages.majority_pools(records, k=2, n=5)
    sources = {r.source for pool in pools for r in pool["records"]}
    assert sources == {"visionarena", "rlhf_v"}


def test_fewshot_split_keeps_room_for_sources_without_annotator_ids():
    dense = [make_record(f"a{i}", source="visionarena", phash=f"a{i}",
                         instruction=f"aq{i}",
                         annotations=[Annotation(f"ann{i % 4}", "A")])
             for i in range(200)]
    other = [make_record(f"o{i}", source="rlhf_v", phash=f"o{i}",
                         instruction=f"oq{i}", real_annotator_ids=False)
             for i in range(200)]

    splits = stages.split_groups(dense + other, seed=42, k=2)
    fewshot_sources = {r.source for r in splits["fewshot"]}
    assert fewshot_sources == {"visionarena", "rlhf_v"}, (
        f"fewshot drew from {fewshot_sources} only")


def test_majority_pools_refuse_an_impossible_request():
    with pytest.raises(ValueError):
        stages.majority_pools([make_record("u0")], k=2, n=5)


def test_annotator_capacity_counts_by_k():
    records = [make_record(f"u{i}", annotations=[Annotation("a", "A")])
               for i in range(3)]
    capacity = stages.annotator_capacity(records)
    assert capacity[1] == 1 and capacity[2] == 1 and capacity[5] == 0


def test_placeholder_annotator_ids_do_not_feed_ind_pools():
    # VisIT-Bench numbers its five judgments positionally; judgment_0 is a
    # different person on every tuple, so it must not look like one annotator
    # with hundreds of labels.
    records = [make_record(f"u{i}", real_annotator_ids=False,
                           annotations=[Annotation("visit_bench_judgment_0", "A")])
               for i in range(20)]
    assert stages.individual_pools(records, k=2, n=3) == []
    assert stages.annotator_capacity(records) == {1: 0, 2: 0, 3: 0, 5: 0}


def test_real_annotator_ids_still_feed_ind_pools():
    records = [make_record(f"u{i}", real_annotator_ids=True,
                           annotations=[Annotation("user_hash_abc", "A")])
               for i in range(20)]
    pools = stages.individual_pools(records, k=2, n=3)
    assert [p["annotator"] for p in pools] == ["user_hash_abc"]


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_export_preference_index_matches_the_label():
    for label, expected in (("A", 1), ("B", 2)):
        instance = to_instance(make_record("u", label=label))
        assert set(instance["preferences"].values()) == {expected}
        assert instance["outputs"] == ["response one", "response two"]


def test_export_uses_real_annotator_ids_when_present():
    record = make_record("u", label="A",
                         annotations=[Annotation("w1", "A"), Annotation("w2", "B")])
    instance = to_instance(record)
    assert instance["preferences"] == {"w1": 1, "w2": 2}


def test_export_of_an_explicit_tie_has_no_forced_choice():
    # A VisionArena "tie (bothbad)" vote is not a hidden A or B. Inventing a
    # preference for it would corrupt the set that exists to test whether
    # confidence falls to 0.5 under genuine indifference.
    record = make_record("t", is_tie=True,
                         annotations=[Annotation("user_a", "tie")])
    instance = to_instance(record)
    assert instance["preferences"] == {}
    assert instance["source"]["is_tie"] is True


def test_export_of_a_split_vote_tie_keeps_the_individual_votes():
    record = make_record("t", is_tie=True,
                         annotations=[Annotation("w1", "A"), Annotation("w2", "B")])
    assert to_instance(record)["preferences"] == {"w1": 1, "w2": 2}


def test_export_rejects_an_unlabelled_non_tie_record():
    record = make_record("bad")
    record.label = None
    with pytest.raises(ValueError, match="no usable label"):
        to_instance(record)


def test_export_carries_provenance_for_the_bias_slices():
    instance = to_instance(make_record("u"))
    for key in ("uid", "dataset", "group_id", "label_origin", "swapped", "is_tie"):
        assert key in instance["source"]


# --------------------------------------------------------------------------
# JudgeAnything parsing
# --------------------------------------------------------------------------

def test_parse_battle_reads_the_winner():
    from prepare_data.adapters.judge_anything import parse_battle
    assert parse_battle("VideoLlama2 vs Baichuan-Omni-1.5 VideoLlama2 wins") == (
        "VideoLlama2", "Baichuan-Omni-1.5", "VideoLlama2")
    assert parse_battle("CoDi vs OneLLM OneLLM wins") == ("CoDi", "OneLLM", "OneLLM")


def test_parse_battle_handles_hyphenated_and_dotted_names():
    # "Baichuan-Omni-1.5" and "Next-GPT" break any whitespace-splitting parse.
    from prepare_data.adapters.judge_anything import parse_battle
    assert parse_battle("Next-GPT vs Baichuan-Omni-1.5 Next-GPT wins") == (
        "Next-GPT", "Baichuan-Omni-1.5", "Next-GPT")


@pytest.mark.parametrize("text", [
    "Unified-IO2 vs ModaVerse",            # no outcome recorded
    "Unified-IO2 vs ModaVerse draw",       # explicit draw
])
def test_parse_battle_returns_no_winner_for_a_tie(text):
    from prepare_data.adapters.judge_anything import parse_battle
    a, b, winner = parse_battle(text)
    assert (a, b) == ("Unified-IO2", "ModaVerse") and winner is None


@pytest.mark.parametrize("text", ["nonsense", "Foo vs Bar Foo wins", ""])
def test_parse_battle_rejects_unknown_models(text):
    from prepare_data.adapters.judge_anything import parse_battle
    assert parse_battle(text) is None


def test_answer_text_flattens_every_shape_seen():
    from prepare_data.adapters.judge_anything import answer_text
    assert answer_text({"type": "text", "content": "hello"}) == "hello"
    assert answer_text({"type": "Text", "content": "hello"}) == "hello"      # case varies
    assert answer_text([{"type": "text", "content": "hello"}]) == "hello"    # list form
    assert answer_text({"type": "text", "content": ["a", "b"]}) == "a\nb"
    assert answer_text(None) == ""
    # A generated-media answer has nothing a text judge can compare.
    assert answer_text({"type": "image", "content": "x.png"}) == ""


def test_image_paths_parses_both_shapes():
    from prepare_data.adapters.judge_anything import image_paths
    assert image_paths("images/a.jpg") == ["images/a.jpg"]
    assert image_paths("['images/a.jpg', 'images/b.jpg']") == [
        "images/a.jpg", "images/b.jpg"]
    assert image_paths(["images/a.jpg"]) == ["images/a.jpg"]
    assert image_paths("") == []


# --------------------------------------------------------------------------
# compatibility with the judge scripts
# --------------------------------------------------------------------------

def test_fewshot_examples_render_without_a_human_key():
    # The judges used to read example["preferences"]["human"] directly. The
    # multimodal splits key preferences by real annotator id so that Ind. pools
    # can group by annotator, which made that lookup a KeyError.
    from open_cascade.data import preferred_index

    by_annotator = to_instance(make_record(
        "u", label="B", annotations=[Annotation("arena_user_xyz", "B")]))
    assert preferred_index(by_annotator) == 2

    aggregate = to_instance(make_record("v", label="A"))   # {"human": 1}
    assert preferred_index(aggregate) == 1


def test_preferred_index_takes_the_majority():
    from open_cascade.data import preferred_index

    record = make_record("u", label="A", annotations=[
        Annotation("w1", "A"), Annotation("w2", "A"), Annotation("w3", "B")])
    assert preferred_index(to_instance(record)) == 1


def test_preferred_index_rejects_a_tie_used_as_a_demonstration():
    from open_cascade.data import preferred_index

    tie = to_instance(make_record("t", is_tie=True,
                                  annotations=[Annotation("u1", "tie")]))
    with pytest.raises(ValueError, match="no preferences"):
        preferred_index(tie)


# --------------------------------------------------------------------------
# raw inspection
# --------------------------------------------------------------------------

def test_quantiles_are_monotonic_despite_missing_values():
    # NaN compares False against everything, so sorting a list containing it
    # yields a non-monotonic order and nonsense quantiles.
    from prepare_data.inspect_raw import _quantiles

    values = [float(v) for v in range(1, 101)] + [float("nan")] * 5
    summary = _quantiles(values)
    assert summary["missing"] == 5
    assert (summary["min"] <= summary["p25"] <= summary["median"]
            <= summary["p75"] <= summary["max"])


def test_quantiles_handles_an_all_missing_column():
    from prepare_data.inspect_raw import _quantiles
    assert _quantiles([float("nan"), float("nan")]) == {}
    assert _quantiles([]) == {}
