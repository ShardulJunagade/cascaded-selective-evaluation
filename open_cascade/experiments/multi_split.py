"""Repeated, group-safe calibration/test draws over cached eval-pool judgements."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import random
from statistics import mean
from typing import Dict, List, Tuple

from open_cascade.data import sample_key
from open_cascade.experiments.single_split import calibrate_and_evaluate


@lru_cache(maxsize=None)
def _image_phash(path: str) -> str:
    """Recover the preparation pipeline's perceptual identity for older exports."""
    from PIL import Image
    from prepare_data.media import perceptual_hash

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Cannot verify leakage groups: image {path} is missing and the export "
            "has no source.media_hash. Restore media or rebuild the export."
        )
    with Image.open(image_path) as image:
        return perceptual_hash(image.convert("RGB"))


def _components(rows: List[Dict]) -> List[List[str]]:
    """Join rows sharing a prompt group or perceptually identical image."""
    parent: Dict[str, str] = {}

    def root(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for row in rows:
        source = row.get("source") or {}
        group = source.get("group_id")
        image_path = row.get("image_path")
        if not group or not image_path:
            raise ValueError("eval_pool rows need source.group_id and image_path")
        group_node = f"group:{group}"
        hashes = source.get("media_hash")
        if not hashes:
            hashes = [_image_phash(image_path)]
        if not isinstance(hashes, list) or not hashes:
            raise ValueError("source.media_hash must be a nonempty list")
        for image_hash in hashes:
            union(group_node, f"image:{image_hash}")

    by_component: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        by_component[root(f"group:{row['source']['group_id']}")].append(sample_key(row))
    return [by_component[key] for key in sorted(by_component)]


def align_pool(pool: Dict[str, List[Dict]], judges: List[str],
               raw_rows: List[Dict] = None) -> Tuple[Dict[str, Dict[str, Dict]], List[List[str]], Dict]:
    """Use only instances scored by every judge; fail on mismatched identities."""
    indexed = {}
    for judge in judges:
        if judge not in pool:
            raise KeyError(f"Missing eval_pool judgements for {judge}")
        rows = {}
        for row in pool[judge]:
            key = sample_key(row)
            if key in rows:
                raise ValueError(f"Duplicate judgement for {judge}: {key[:100]}")
            if not row.get("preferences") or len(row.get("probs", [])) != 2:
                raise ValueError(f"Invalid labelled judgement for {judge}: {key[:100]}")
            rows[key] = row
        indexed[judge] = rows

    common = set.intersection(*(set(indexed[judge]) for judge in judges))
    if not common:
        raise ValueError("No eval_pool instances were scored by every judge")
    first = indexed[judges[0]]
    ordered = [key for key in first if key in common]
    for key in ordered:
        for judge in judges[1:]:
            if indexed[judge][key]["preferences"] != first[key]["preferences"]:
                raise ValueError(f"Human labels differ across judges for {key[:100]}")
    # Build connectivity over the entire export, including rows omitted by one
    # judge. An omitted row can bridge two scored rows through different hashes.
    full_components = _components(raw_rows if raw_rows is not None else [first[key] for key in ordered])
    components = [[key for key in group if key in common] for group in full_components]
    components = [group for group in components if group]
    counts = {judge: len(indexed[judge]) - len(common) for judge in judges}
    return indexed, components, {"n_common": len(common), "excluded_by_judge": counts,
                                 "n_components": len(components)}


def validate_scored_pool(raw_rows: List[Dict], pool: Dict[str, List[Dict]], judges: List[str]) -> None:
    """Reject stale cache rows or labels from a different export."""
    raw = {}
    for row in raw_rows:
        key = sample_key(row)
        if key in raw:
            raise ValueError(f"Duplicate instance in eval_pool: {key[:100]}")
        raw[key] = row
    if not raw:
        raise ValueError("eval_pool is empty")
    for judge in judges:
        for row in pool[judge]:
            key = sample_key(row)
            if key not in raw:
                raise ValueError(f"{judge} cache has a row outside the current eval_pool: {key[:100]}")
            source = raw[key]
            if (row.get("preferences") != source.get("preferences")
                    or row.get("source", {}).get("group_id") != source.get("source", {}).get("group_id")):
                raise ValueError(f"{judge} cache disagrees with eval_pool labels/groups: {key[:100]}")


def draw_keys(components: List[List[str]], calibration_size: int, seed: int) -> Tuple[List[str], List[str]]:
    if calibration_size <= 0 or calibration_size >= sum(map(len, components)):
        raise ValueError("calibration_size must be between zero and the common pool size")
    groups = list(components)
    random.Random(seed).shuffle(groups)
    calibration, test = [], []
    for index, group in enumerate(groups):
        if len(calibration) < calibration_size and index < len(groups) - 1:
            calibration.extend(group)
        else:
            test.extend(group)
    if not calibration or not test:
        raise ValueError("Cannot draw a nonempty calibration/test split over leakage groups")
    return calibration, test


def run_guarantee_success(config, pool: Dict[str, List[Dict]], raw_rows=None) -> Dict:
    """Measure how often the target agreement holds across group-safe random splits."""
    if config.evaluate.n_splits <= 0:
        raise ValueError("evaluate.n_splits must be positive")
    indexed, components, inventory = align_pool(pool, config.judges, raw_rows)
    total = inventory["n_common"]
    if config.data.calibration_size >= total:
        raise ValueError(f"calibration_size={config.data.calibration_size} needs fewer than {total} rows")

    rows = []
    for number in range(config.evaluate.n_splits):
        seed = config.evaluate.seed + number
        cal_keys, test_keys = draw_keys(components, config.data.calibration_size, seed)
        calibration = {judge: [indexed[judge][key] for key in cal_keys] for judge in config.judges}
        test = {judge: [indexed[judge][key] for key in test_keys] for judge in config.judges}
        result = calibrate_and_evaluate(config, calibration, test, config.cascade.alpha)
        met = result.coverage > 0 and result.human_agreement >= 1 - config.cascade.alpha
        rows.append({"seed": seed, "n_calibration": len(cal_keys), "n_test": len(test_keys),
                     "agreement": result.human_agreement, "coverage": result.coverage,
                     "met_target": met, "lambda_hats": result.summary()["lambda_hats"]})

    summary = {
        "alpha": config.cascade.alpha,
        "delta": config.cascade.delta,
        "target_agreement": 1 - config.cascade.alpha,
        "n_splits": len(rows),
        "guarantee_success_rate": mean(row["met_target"] for row in rows),
        "mean_agreement": mean(row["agreement"] for row in rows),
        "mean_coverage": mean(row["coverage"] for row in rows),
        **inventory,
        "splits": rows,
    }
    print(f"Common scored rows: {total}; leakage groups: {inventory['n_components']}")
    print(f"Guarantee success rate: {summary['guarantee_success_rate']:.3f} "
          f"({sum(row['met_target'] for row in rows)}/{len(rows)} splits)")
    print(f"Mean agreement: {summary['mean_agreement']:.4f}; "
          f"mean coverage: {summary['mean_coverage']:.4f}")
    return summary
