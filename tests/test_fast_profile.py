from collections import Counter

from cluster.build_fast_profile import select_component_stratified, source_name
from open_cascade.data import sample_key


def row(index, source, group=None, image_hash=None):
    return {
        "instruction": f"question {index}",
        "outputs": [f"a{index}", f"b{index}"],
        "preferences": {"human": 1},
        "image_path": f"image-{index}.jpg",
        "source": {
            "dataset": source,
            "group_id": group or f"group-{index}",
            "media_hash": [image_hash or f"hash-{index}"],
        },
    }


def test_fast_selection_is_deterministic_stratified_and_component_safe():
    rows = [row(i, "large") for i in range(80)]
    rows += [row(i + 80, "small") for i in range(20)]
    # These two must travel together even though they have different group ids.
    rows[0]["source"]["media_hash"] = ["shared"]
    rows[1]["source"]["media_hash"] = ["shared"]

    first = select_component_stratified(rows, target=50, seed=42)
    second = select_component_stratified(rows, target=50, seed=42)
    assert [sample_key(item) for item in first] == [sample_key(item) for item in second]
    assert len(first) == 50

    keys = {sample_key(item) for item in first}
    assert (sample_key(rows[0]) in keys) == (sample_key(rows[1]) in keys)
    assert Counter(source_name(item) for item in first) == {"large": 40, "small": 10}
