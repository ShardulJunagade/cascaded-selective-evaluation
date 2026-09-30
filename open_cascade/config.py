"""Experiment configuration.

One experiment = one YAML file in `configs/`. The file is loaded into the dataclasses below,
so every setting has a name, a type and a default in exactly one place. Unknown keys raise
an error, so a typo such as `alhpa: 0.1` fails loudly instead of being silently ignored.

Any value can be overridden from the command line without editing the file:

    python -m open_cascade evaluate configs/text_mistral_qwen7b.yaml --set cascade.alpha=0.1
"""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


@dataclass
class DataConfig:
    """Where the dataset comes from and how it is split (`prepare-data`)."""

    builder: str                    # key in open_cascade.builders.BUILDERS
    out_dir: str                    # splits are written to {out_dir}/split/
    n_annotators: int = 3           # N simulated annotators (paper S2.2)
    k_shot: int = 3                 # K few-shot examples per simulated annotator
    calibration_size: int = 500
    seed: int = 42
    options: Dict[str, Any] = field(default_factory=dict)  # builder-specific settings

    @property
    def split_dir(self) -> Path:
        return Path(self.out_dir) / "split"

    @property
    def fewshot_file(self) -> Path:
        return self.split_dir / "fewshot.jsonl"

    def split_file(self, split: str) -> Path:
        return self.split_dir / f"{split}.jsonl"


@dataclass
class ScoringConfig:
    """How judges are run over the splits on the GPU (`score`)."""

    splits: List[str] = field(default_factory=lambda: ["calibration", "test"])
    chunk_size: int = 64            # samples per vLLM call; each expands to 2N prompts
    max_fewshot_examples: Optional[int] = None  # cap on K at scoring time; None = use all
    prefix_caching: bool = True
    resume: bool = True             # continue a partly written result file
    released_mistral_compat: bool = False  # text only: the released generate-and-parse path


@dataclass
class CascadeConfig:
    """How the cascade is calibrated (`evaluate`)."""

    alpha: float = 0.15             # risk tolerance; target human agreement is 1 - alpha
    delta: float = 0.1              # the guarantee holds with probability 1 - delta
    split_delta: bool = True        # False = full delta per judge, as in the released code
    threshold_method: str = "fixed_sequence_testing"  # key in thresholds.THRESHOLD_METHODS


@dataclass
class EvaluateConfig:
    """Which experiments `evaluate` runs, and where their results go."""

    experiments: List[str] = field(default_factory=lambda: ["cascade"])
    alphas: List[float] = field(default_factory=lambda: [0.30, 0.25, 0.20, 0.15, 0.10, 0.05])
    output_dir: str = "./outputs"


@dataclass
class ValidationConfig:
    """Thresholds for `validate`: a re-scored judge vs. released judgements."""

    released_file: str = "./result/mistral-7b-instruct.calibration.jsonl"
    split: str = "calibration"
    label_agreement_threshold: float = 0.94
    correlation_threshold: float = 0.90
    human_agreement_tolerance: float = 0.02
    strict_confidence: bool = False  # also require the confidence correlation threshold
    debug_examples: int = 8          # print the largest confidence shifts and label flips


@dataclass
class ExperimentConfig:
    name: str
    judges: List[str]               # weakest (cheapest) first
    result_dir: str                 # judgements live at {result_dir}/{judge}.{split}.jsonl
    data: DataConfig
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    evaluate: EvaluateConfig = field(default_factory=EvaluateConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def result_file(self, judge: str, split: str) -> Path:
        return Path(self.result_dir) / f"{judge}.{split}.jsonl"

    def to_yaml(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)


# section name -> dataclass, for the nested parts of ExperimentConfig
_SECTIONS = {
    "data": DataConfig,
    "scoring": ScoringConfig,
    "cascade": CascadeConfig,
    "evaluate": EvaluateConfig,
    "validation": ValidationConfig,
}


def load_config(path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load a YAML experiment config, applying `key.subkey=value` overrides first."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    for override in overrides:
        apply_override(raw, override)

    raw.setdefault("name", path.stem)
    return config_from_dict(raw, where=str(path))


def config_from_dict(raw: Dict[str, Any], where: str = "config") -> ExperimentConfig:
    raw = dict(raw)
    if "data" not in raw:
        raise ValueError(f"{where}: missing the required `data` section.")
    for section, cls in _SECTIONS.items():
        if section in raw:
            raw[section] = _make(cls, raw[section] or {}, f"{where} [{section}]")
    return _make(ExperimentConfig, raw, where)


def apply_override(raw: Dict[str, Any], override: str) -> None:
    """Apply `cascade.alpha=0.1` to a raw config dict. The value is parsed as YAML, so
    numbers, booleans and lists (`judges=[a,b]`) all work."""
    key, sep, value = override.partition("=")
    if not sep or not key:
        raise ValueError(f"Override '{override}' should look like section.key=value")

    *parents, last = key.split(".")
    node = raw
    for parent in parents:
        node = node.setdefault(parent, {})
    node[last] = yaml.safe_load(value)


def _make(cls, values: Dict[str, Any], where: str):
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}")
    try:
        return cls(**values)
    except TypeError as error:  # a required key is missing
        raise ValueError(f"{where}: {error}") from None
