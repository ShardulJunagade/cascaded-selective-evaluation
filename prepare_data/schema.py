"""The canonical record every adapter emits, plus its validator.

One record is one *pairwise preference instance*: a piece of media, an
instruction, two candidate responses and a human label. Everything the bias
tables and the abstention-vs-IAA analysis need is carried on the record itself
so that no stage has to reach back into a raw source.

Label convention
----------------
`label` is "A" or "B" and always refers to the *current* contents of
`response_a` / `response_b`. The position-randomization stage swaps the two
responses and flips the label together, setting `swapped=True`. Downstream code
therefore never needs to know whether a record was swapped in order to read the
label correctly; `swapped` exists for auditing and for the position-bias table.

This is the fix for the defect in the old `prepare_vlm_dataset.py`, which wrote
`outputs=[chosen, rejected]` with `preferences={"human": 1}` for every row - a
constant label that any judge answering "A" every time would match perfectly.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

LABELS = ("A", "B")
MODALITIES = ("image", "video", "audio")
LABEL_ORIGINS = ("vote", "majority", "ranking_derived", "correction")

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercased, whitespace-collapsed form used for dedup and equality checks."""
    return _WHITESPACE.sub(" ", (text or "").strip().lower())


def make_uid(source: str, *parts: Any) -> str:
    """Stable id: same inputs give the same uid on every run and every machine."""
    payload = "\x1f".join([source, *(str(p) for p in parts)])
    return f"{source}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def make_group_id(source: str, media_hash: List[str], instruction: str) -> str:
    """Leakage unit: same image(s) + same instruction means the same group.

    Splits are drawn over groups so that two pairs built from one prompt can
    never land on opposite sides of a calibration/test boundary.
    """
    payload = "\x1f".join([source, *sorted(media_hash), normalize_text(instruction)])
    return f"g-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class Annotation:
    annotator: str
    label: str            # "A" | "B" | "tie"

    def to_dict(self) -> Dict[str, str]:
        return {"annotator": self.annotator, "label": self.label}


@dataclass
class Record:
    uid: str
    source: str
    group_id: str
    modality: str                        # image | video | audio

    media: List[str]                     # paths relative to data/vlm_v2/
    media_hash: List[str]                # perceptual hash, one per media item

    instruction: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str

    label: Optional[str]                 # "A" | "B"; None only for ties
    label_origin: str                    # vote | majority | ranking_derived | correction

    annotations: List[Dict[str, str]] = field(default_factory=list)
    n_ann: int = 0
    agreement: Optional[float] = None    # fraction agreeing with the majority
    is_tie: bool = False
    swapped: bool = False

    context: List[Dict[str, Any]] = field(default_factory=list)   # empty in the main protocol
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Record":
        known = {f for f in cls.__dataclass_fields__}      # noqa: F821 - dataclass attr
        return cls(**{k: v for k, v in data.items() if k in known})


class ValidationError(ValueError):
    """A record violated the schema. Carries the uid so the funnel can log it."""


def validate(record: Record, *, strict_media: bool = True) -> None:
    """Raise ValidationError if a record cannot be used downstream."""
    problems: List[str] = []

    if not record.uid:
        problems.append("empty uid")
    if record.modality not in MODALITIES:
        problems.append(f"modality {record.modality!r} not in {MODALITIES}")
    if record.label_origin not in LABEL_ORIGINS:
        problems.append(f"label_origin {record.label_origin!r} not in {LABEL_ORIGINS}")

    if not record.instruction.strip():
        problems.append("empty instruction")
    if not record.response_a.strip() or not record.response_b.strip():
        problems.append("empty response")

    if record.is_tie:
        if record.label is not None:
            problems.append("tie record carries a label")
    else:
        if record.label not in LABELS:
            problems.append(f"label {record.label!r} not in {LABELS}")

    if strict_media:
        if not record.media:
            problems.append("no media")
        if len(record.media) != len(record.media_hash):
            problems.append("media/media_hash length mismatch")

    if record.n_ann != len(record.annotations):
        problems.append(f"n_ann={record.n_ann} but {len(record.annotations)} annotations")
    if record.agreement is not None and not 0.0 <= record.agreement <= 1.0:
        problems.append(f"agreement {record.agreement} outside [0, 1]")

    if problems:
        raise ValidationError(f"{record.uid}: " + "; ".join(problems))


def aggregate_annotations(annotations: List[Annotation]) -> Dict[str, Any]:
    """Majority label, agreement level and tie status for a set of annotations.

    A tie is either an explicit "tie" majority or an exact A/B split. Ties are
    routed out of the main protocol because a forced-choice risk is undefined
    when the annotator is indifferent.
    """
    if not annotations:
        return {"label": None, "agreement": None, "is_tie": True, "n_ann": 0}

    counts: Dict[str, int] = {}
    for annotation in annotations:
        counts[annotation.label] = counts.get(annotation.label, 0) + 1

    top = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == top)
    total = len(annotations)

    if len(winners) > 1 or winners[0] == "tie":
        return {"label": None, "agreement": top / total, "is_tie": True, "n_ann": total}

    return {"label": winners[0], "agreement": top / total, "is_tie": False, "n_ann": total}
