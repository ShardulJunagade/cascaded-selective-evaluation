"""Dataset builders, used by `python -m open_cascade prepare-data`.

    base.py            DatasetBuilder -- common instance format, splitting, writing
    released_text.py   the paper's text data, rebuilt from ./result/
    rlhf_v.py          openbmb/RLHF-V-Dataset (VLM)

To add a dataset: create a new file with a DatasetBuilder subclass, add it to BUILDERS
below, and point an experiment config's `data.builder` at the new key.
"""
from open_cascade.builders.base import DatasetBuilder
from open_cascade.builders.released_text import ReleasedTextBuilder
from open_cascade.builders.rlhf_v import RLHFVBuilder
from open_cascade.config import DataConfig

# data.builder -> builder class
BUILDERS = {
    "released_text": ReleasedTextBuilder,
    "rlhf_v": RLHFVBuilder,
}


def get_builder(config: DataConfig) -> DatasetBuilder:
    if config.builder not in BUILDERS:
        raise KeyError(f"Unknown dataset builder '{config.builder}'. Known: {sorted(BUILDERS)}")
    return BUILDERS[config.builder](config, **config.options)
