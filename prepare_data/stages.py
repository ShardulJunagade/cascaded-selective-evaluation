"""Shared stages applied identically to every source.

Order matters and is fixed:

    hygiene -> dedup -> tie routing -> position randomization -> group split

Position randomization runs *after* tie routing (ties carry no label to flip)
and *before* splitting, so that the label balance reported per split is the
balance the judge will actually see.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from prepare_data.config import (CONTEXT_LIMIT_TOKENS, FEWSHOT_DENSITY_SHARE,
                                 IMAGE_TOKEN_ESTIMATE, K_MAX, SOURCE_PRIORITY,
                                 SPLIT_FRACTIONS)
from prepare_data.funnel import Funnel
from prepare_data.schema import Record, normalize_text

# --------------------------------------------------------------------------
# 1. Hygiene
# --------------------------------------------------------------------------

REFUSAL_PATTERNS = [
    r"^i'?m sorry",
    r"^i am sorry",
    r"^sorry, (?:i|but)",
    r"^i (?:can'?t|cannot|won'?t) (?:help|assist|provide|answer|comply)",
    r"^as an ai(?: language model)?, i (?:can'?t|cannot)",
    r"^i'?m (?:not able|unable) to",
    r"^unfortunately, i (?:can'?t|cannot)",
]
_REFUSAL = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.match((text or "").strip()))


def load_judge_tokenizer(model: Optional[str] = None):
    """The judge's own tokenizer, for exact token counts. None if unavailable.

    Optional on purpose: pulling the Qwen processor is a model download, and
    the count only drives a *flag*, never a drop, so the character estimate is
    enough for a first pass.
    """
    from prepare_data.config import JUDGE_MODEL

    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model or JUDGE_MODEL)
        return getattr(processor, "tokenizer", processor)
    except Exception:                              # noqa: BLE001 - optional path
        return None


def estimate_tokens(record: Record, k: int = K_MAX, tokenizer=None) -> int:
    """Prompt size for a K-shot judge call, including image tokens.

    With `tokenizer` the text is counted exactly; without it, at ~4 characters
    per token. Image tokens are estimated either way, since the true count
    depends on the judge's `max_pixels` at prompt time.
    """
    text = f"{record.instruction}\n{record.response_a}\n{record.response_b}"
    if tokenizer is not None:
        text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    else:
        text_tokens = len(text) / 4
    per_example = text_tokens + IMAGE_TOKEN_ESTIMATE * len(record.media)
    return int(per_example * (k + 1))


def hygiene(records: Iterable[Record], funnel: Funnel,
            stage: str = "hygiene", tokenizer=None) -> List[Record]:
    kept: List[Record] = []
    for record in records:
        a, b = record.response_a, record.response_b

        if not a.strip() or not b.strip():
            funnel.drop(stage, record.source, "empty response")
            continue
        if normalize_text(a) == normalize_text(b):
            funnel.drop(stage, record.source, "identical responses")
            continue
        if is_refusal(a) and is_refusal(b):
            funnel.drop(stage, record.source, "both responses are refusals")
            continue

        tokens = estimate_tokens(record, tokenizer=tokenizer)
        record.meta["est_tokens_k_max"] = tokens
        record.meta["tokens_exact"] = tokenizer is not None
        record.meta["over_context_budget"] = tokens > CONTEXT_LIMIT_TOKENS
        if record.meta["over_context_budget"]:
            # Flagged, not dropped: the judge can fall back to K=0 or K=1 for
            # these, and the count is evidence for the context-length bottleneck.
            funnel.drop(stage, record.source, "flagged: over context budget (kept)")

        kept.append(record)
        funnel.keep(stage, record.source)
    return kept


# --------------------------------------------------------------------------
# 2. Cross-source dedup
# --------------------------------------------------------------------------

def _priority(source: str) -> int:
    return SOURCE_PRIORITY.index(source) if source in SOURCE_PRIORITY else len(SOURCE_PRIORITY)


def _image_key(record: Record) -> str:
    return "|".join(sorted(record.media_hash))


def _prompt_key(record: Record) -> Tuple[str, str]:
    return (_image_key(record), normalize_text(record.instruction))


def _instance_key(record: Record) -> Tuple[str, str, str]:
    """Identity of a comparison: image, prompt, and the unordered response pair."""
    responses = tuple(sorted((normalize_text(record.response_a),
                              normalize_text(record.response_b))))
    return (*_prompt_key(record), "\x1f".join(responses))


def dedup(records: Sequence[Record], funnel: Funnel,
          stage: str = "dedup") -> List[Record]:
    """Remove duplicate *comparisons*: same image, same prompt, same response pair.

    The pair is compared unordered, so an A/B flip still counts as the same
    instance. On a collision the record from the higher-priority source wins,
    and within a source the first by uid wins, so the result does not depend on
    input order.

    What this deliberately does **not** do is deduplicate on image + prompt
    alone. Several of these benchmarks ask different questions of the same
    image, and more importantly they pit *different model pairs* against each
    other on one prompt: VisIT-Bench compares seven models per image, and
    JudgeAnything records four battles per query. Those are distinct
    comparisons, not repeats. A prompt-level key removed 22% of the corpus
    within sources, and a further 76 JudgeAnything records across them, for no
    gain - the leakage it would have guarded against is already handled by the
    group-aware split, which merges components on perceptual hash and so keeps
    every record sharing an image on one side of the boundary whatever its
    source.
    """
    ordered = sorted(records, key=lambda r: (_priority(r.source), r.uid))

    seen: Dict[Tuple[str, str, str], Record] = {}
    kept: List[Record] = []
    for record in ordered:
        key = _instance_key(record)
        if key in seen:
            other = seen[key].source
            reason = ("duplicate instance" if other == record.source
                      else f"duplicate instance, already in {other}")
            funnel.drop(stage, record.source, reason)
            continue
        seen[key] = record
        kept.append(record)
        funnel.keep(stage, record.source)
    return kept


# --------------------------------------------------------------------------
# 3. Tie routing
# --------------------------------------------------------------------------

def route_ties(records: Iterable[Record], funnel: Funnel,
               stage: str = "tie_routing") -> Tuple[List[Record], List[Record]]:
    """Split off tied instances.

    A forced-choice risk is undefined when the annotator is indifferent, so
    ties leave the main protocol. They are kept as their own set, which is what
    the "does confidence approach its 0.5 lower bound?" test runs on.
    """
    main: List[Record] = []
    ties: List[Record] = []
    for record in records:
        if record.is_tie or record.label is None:
            ties.append(record)
            funnel.drop(stage, record.source, "routed to tie set")
        else:
            main.append(record)
            funnel.keep(stage, record.source)
    return main, ties


# --------------------------------------------------------------------------
# 4. Position randomization
# --------------------------------------------------------------------------

def randomize_positions(records: Iterable[Record], seed: int = 42,
                        funnel: Optional[Funnel] = None,
                        stage: str = "position_randomized") -> List[Record]:
    """Randomize which response sits in slot A, flipping the label with it.

    This is the fix for the defect in the old `prepare_vlm_dataset.py`, which
    always wrote the preferred response first and labelled every instance the
    same way. Under that layout a judge that answers "A" every time scores
    100%, so the reported agreement measured nothing.

    The coin is seeded by uid rather than by position in the list, so a record
    gets the same assignment no matter which sources ran or in what order.
    """
    out: List[Record] = []
    for record in records:
        coin = random.Random(f"{seed}:{record.uid}").random()
        if coin < 0.5:
            record.response_a, record.response_b = record.response_b, record.response_a
            record.model_a, record.model_b = record.model_b, record.model_a
            record.label = {"A": "B", "B": "A"}.get(record.label, record.label)
            record.annotations = [
                {"annotator": a["annotator"],
                 "label": {"A": "B", "B": "A"}.get(a["label"], a["label"])}
                for a in record.annotations
            ]
            record.meta["len_a"], record.meta["len_b"] = (
                record.meta.get("len_b"), record.meta.get("len_a"))
            record.swapped = True
        else:
            record.swapped = False
        out.append(record)
        if funnel is not None:
            funnel.keep(stage, record.source)
    return out


# --------------------------------------------------------------------------
# 5. Group-aware split
# --------------------------------------------------------------------------

class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:      # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def leakage_components(records: Sequence[Record]) -> Dict[str, str]:
    """Map uid -> component id, merging anything that shares a group or an image.

    Splitting on `group_id` alone is not enough: one image can appear under two
    different instructions, and two sources can share an image. Merging group
    ids with perceptual hashes into connected components makes the resulting
    splits disjoint on *both* keys, which is what the leakage assertion checks.
    """
    uf = _UnionFind()
    for record in records:
        node = f"grp:{record.group_id}"
        uf.find(node)
        for phash in record.media_hash:
            uf.union(node, f"img:{phash}")
    return {record.uid: uf.find(f"grp:{record.group_id}") for record in records}


def annotator_density(records: Sequence[Record], k: int) -> Dict[str, int]:
    """Per record, how many of its annotators are prolific enough for an Ind. pool.

    Used only to order the split, never to choose labels.
    """
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        if not has_real_annotator_ids(record):
            continue
        for annotation in record.annotations:
            if annotation["label"] in ("A", "B"):
                counts[annotation["annotator"]] += 1

    prolific = {a for a, n in counts.items() if n >= k + 1}
    density: Dict[str, int] = {}
    for record in records:
        if not has_real_annotator_ids(record):
            density[record.uid] = 0
            continue
        density[record.uid] = sum(
            1 for a in record.annotations if a["annotator"] in prolific)
    return density


def split_groups(records: Sequence[Record], seed: int = 42,
                 fractions: Optional[Dict[str, float]] = None,
                 funnel: Optional[Funnel] = None,
                 k: int = K_MAX) -> Dict[str, List[Record]]:
    """Partition records into fewshot / dev / eval_pool over leakage components.

    No single calibration/test split is written. The evaluation script draws
    many random cal/test splits from `eval_pool`, because one split cannot show
    whether the (1 - delta) guarantee holds.

    Components whose records come from prolific annotators are offered to the
    fewshot split first. Without that, a purely random assignment scatters each
    annotator's votes across all three splits: an annotator with ten votes
    corpus-wide expects one inside a 10% fewshot split, so almost nobody clears
    the K+1 bar *within* the split and the Ind. pools come out empty even
    though thousands of annotators qualify overall. Concentrating them costs
    nothing in validity, because fewshot is prompt material and is never
    evaluated on.
    """
    fractions = fractions or SPLIT_FRACTIONS
    components = leakage_components(records)

    by_component: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        by_component[components[record.uid]].append(record)

    names = list(fractions)
    order = sorted(by_component)
    random.Random(f"{seed}-split").shuffle(order)

    density = annotator_density(records, k=k)
    component_density = {
        component: sum(density.get(r.uid, 0) for r in rows)
        for component, rows in by_component.items()
    }

    # A perceptual hash collision on a near-blank image would chain unrelated
    # records into one giant component, which would then have to land wholly in
    # one split and wreck the fractions. Surface it rather than let it pass.
    if funnel is not None and by_component:
        largest = max(len(rows) for rows in by_component.values())
        funnel.note(
            f"split: {len(by_component)} leakage components over {len(records)} records; "
            f"largest holds {largest} ({100.0 * largest / len(records):.1f}%)."
        )
        if largest > 0.2 * len(records):
            funnel.note(
                "split: WARNING - one component holds over 20% of the data, which "
                "usually means a perceptual-hash collision chained unrelated images. "
                "Inspect before trusting the split fractions."
            )

    # Assign whole components, largest-remainder style, so that the row counts
    # land near the requested fractions even though components vary in size.
    total_rows = sum(len(by_component[c]) for c in order)
    quotas = {name: fractions[name] * total_rows for name in names}
    assigned: Dict[str, List[Record]] = {name: [] for name in names}

    # Part of the fewshot split is filled first from the components richest in
    # prolific annotators, so that Ind. pools have annotators clearing K+1
    # *inside* the split. Only part: annotator ids exist in one source only, so
    # filling the whole quota this way makes fewshot ~100% VisionArena, and the
    # Maj. pools drawn from it would demonstrate a single dataset's style to a
    # judge that has to grade four. The rest of the quota is left to the
    # ordinary deficit pass, which draws from every source.
    remaining = list(order)
    if "fewshot" in names:
        density_budget = FEWSHOT_DENSITY_SHARE * quotas["fewshot"]
        remaining.sort(key=lambda component: -component_density[component])
        still: List[str] = []
        for component in remaining:
            rows = by_component[component]
            if (component_density[component] > 0
                    and len(assigned["fewshot"]) + len(rows) <= density_budget):
                assigned["fewshot"].extend(rows)
            else:
                still.append(component)
        remaining = still
        # Drop back to the seeded random order for the rest, so the evaluation
        # splits are not ordered by annotator prolificacy.
        random.Random(f"{seed}-split-rest").shuffle(remaining)

    for component in remaining:
        rows = by_component[component]
        deficits = {name: quotas[name] - len(assigned[name]) for name in names}
        target = max(names, key=lambda n: (deficits[n], n))
        assigned[target].extend(rows)

    if funnel is not None:
        for name, rows in assigned.items():
            for record in rows:
                funnel.keep(f"split:{name}", record.source)

    assert_no_leakage(assigned)
    return assigned


def assert_no_leakage(splits: Dict[str, List[Record]]) -> None:
    """Fail loudly if any group id or image crosses a split boundary."""
    seen_groups: Dict[str, str] = {}
    seen_images: Dict[str, str] = {}

    for name, records in splits.items():
        for record in records:
            other = seen_groups.setdefault(record.group_id, name)
            if other != name:
                raise AssertionError(
                    f"group_id {record.group_id} appears in both {other} and {name}")
            for phash in record.media_hash:
                other = seen_images.setdefault(phash, name)
                if other != name:
                    raise AssertionError(
                        f"image {phash} appears in both {other} and {name}")


# --------------------------------------------------------------------------
# 6. Few-shot pools for Simulated Annotators
# --------------------------------------------------------------------------

def has_real_annotator_ids(record: Record) -> bool:
    """Whether this record's annotator names identify actual people.

    VisIT-Bench ships five indistinguishable judgment rows per tuple with no
    worker id, so the adapter numbers them positionally. Those numbers are not
    annotator identity: `visit_bench_judgment_0` is a different person on every
    tuple, and treating it as one would build an Ind. pool out of a fiction.
    """
    return bool(record.meta.get("real_annotator_ids"))


def individual_pools(records: Sequence[Record], k: int,
                     n: int, seed: int = 42) -> List[Dict]:
    """Simulated Annotators (Ind.): one K-shot pool per *real* annotator.

    An annotator needs at least K+1 labels to be usable - K go into the prompt
    and at least one must be left out - so anyone below that is dropped. On
    these sources that cuts deeply, which is the "too few annotators" bottleneck.
    """
    by_annotator: Dict[str, List[Record]] = defaultdict(list)
    for record in records:
        if not has_real_annotator_ids(record):
            continue
        for annotation in record.annotations:
            if annotation["label"] in ("A", "B"):
                by_annotator[annotation["annotator"]].append(record)

    usable = {a: rows for a, rows in by_annotator.items() if len(rows) >= k + 1}
    rng = random.Random(f"{seed}-ind-{k}-{n}")
    chosen = sorted(usable, key=lambda a: (-len(usable[a]), a))[:n]

    pools = []
    for annotator in chosen:
        rows = sorted(usable[annotator], key=lambda r: r.uid)
        pools.append({"annotator": annotator,
                      "records": rng.sample(rows, k)})
    return pools


def majority_pools(records: Sequence[Record], k: int, n: int,
                   seed: int = 42) -> List[Dict]:
    """Simulated Annotators (Maj.): N disjoint K-sets, stratified by source.

    Demonstrations are drawn round-robin across sources rather than uniformly.
    The corpus is dominated by VisionArena, so a uniform draw would show the
    judge in-the-wild arena chat almost exclusively, while it has to grade
    RLHF-V corrections and MM-RLHF ranking-derived pairs too. Each pool should
    look like the corpus, not like its largest member.
    """
    rng = random.Random(f"{seed}-maj-{k}-{n}")

    by_source: Dict[str, List[Record]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r.uid):
        by_source[record.source].append(record)
    for rows in by_source.values():
        rng.shuffle(rows)

    if len(records) < n * k:
        raise ValueError(
            f"Need N*K={n * k} few-shot records for the Maj. pools, have "
            f"{len(records)}. Lower --N/--K or raise the fewshot split fraction."
        )

    # Round-robin over sources, largest first, so a small source still appears.
    order = sorted(by_source, key=lambda s: (-len(by_source[s]), s))
    interleaved: List[Record] = []
    while len(interleaved) < n * k:
        progressed = False
        for source in order:
            if by_source[source]:
                interleaved.append(by_source[source].pop())
                progressed = True
                if len(interleaved) == n * k:
                    break
        if not progressed:
            break

    return [{"annotator": f"majority_{i}", "records": interleaved[i * k:(i + 1) * k]}
            for i in range(n)]


def annotator_capacity(records: Sequence[Record]) -> Dict[int, int]:
    """How many *real* annotators clear the K+1 bar, for each K.

    Counts only sources with genuine annotator ids, for the same reason
    `individual_pools` does - see `has_real_annotator_ids`. Feeds the bias table.
    """
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        if not has_real_annotator_ids(record):
            continue
        for annotation in record.annotations:
            if annotation["label"] in ("A", "B"):
                counts[annotation["annotator"]] += 1
    return {k: sum(1 for n in counts.values() if n >= k + 1) for k in (1, 2, 3, 5)}
