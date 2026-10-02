"""VisionArena-Battle: in-the-wild arena votes.

Real users upload an image, ask their own question, and vote between responses
from two anonymous VLMs. That makes it the only source with genuine annotator
identity (a hash of the voting user), which is what the Simulated Annotators
(Ind.) pools need - but the votes are heavy-tailed, so most users fall below
the K+1 labels an Ind. pool requires.

Four label values exist: `model_a`, `model_b`, `tie`, `tie (bothbad)`. Both tie
forms are routed to the tie set with the original value kept in
`meta.tie_type`, since "both are good" and "both are bad" are different kinds
of indifference and the confidence -> 0.5 test should be able to separate them.

Only single-turn battles enter the main protocol; multi-turn conversations are
counted and dropped, because the judge prompt takes one instruction.

The dataset is gated. Accept the terms on the dataset page while signed in,
then `hf auth login` before `python -m prepare_data.download --source visionarena`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

from prepare_data.adapters.base import Adapter
from prepare_data.schema import (Annotation, Record, make_group_id, make_uid)

WINNER_TO_LABEL = {"model_a": "A", "model_b": "B"}
TIE_VALUES = {"tie", "tie (bothbad)"}


def turns(conversation: Any) -> List[Dict]:
    """Flatten a conversation field to a list of {role, content} messages.

    VisionArena nests one level deeper than a plain message list: the column
    type is List(List({content, role})), so a single-turn battle arrives as
    [[{user}], [{assistant}]]. Flattening recursively handles both that shape
    and the flat one, so the adapter does not depend on which the release uses.
    """
    if not conversation:
        return []
    if isinstance(conversation, dict):
        return [conversation]

    flat: List[Dict] = []
    for item in conversation:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            flat.extend(turns(item))
    return flat


def first_user_text(conversation: List[Dict]) -> str:
    for turn in conversation:
        if turn.get("role") == "user":
            content = turn.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                return " ".join(p for p in parts if p).strip()
    return ""


def first_assistant_text(conversation: List[Dict]) -> str:
    for turn in conversation:
        if turn.get("role") == "assistant":
            content = turn.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text"]
                return " ".join(p for p in parts if p).strip()
    return ""


def count_exchanges(conversation: List[Dict]) -> int:
    return sum(1 for turn in conversation if turn.get("role") == "user")


def active_categories(categories: Any) -> Optional[str]:
    """`categories` is a dict of booleans (captioning, ocr, diagram, ...).

    Flattened to a sorted comma-joined string so the category distribution in
    the bias tables counts each combination once.
    """
    if not isinstance(categories, dict):
        return str(categories) if categories else None
    active = sorted(key for key, value in categories.items() if value)
    return ",".join(active) if active else "none"


class VisionArenaAdapter(Adapter):
    name = "visionarena"

    def _image(self, row: Dict) -> Optional[Any]:
        """The battle image, wherever this release happens to put it."""
        for field in ("images", "image"):
            value = row.get(field)
            if isinstance(value, list) and value:
                return value[0]
            if value is not None and not isinstance(value, list):
                return value
        # Some releases attach the image to the first user turn instead.
        for turn in turns(row.get("conversation_a")):
            content = turn.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        return part.get("image")
        return None

    def iter_records(self) -> Iterator[Record]:
        dataset = self.load_hf()
        self.funnel.note(
            f"visionarena: {len(dataset)} raw battles. The only source with real "
            "annotator ids (hashed voting user), so it is the basis for the Ind. pools."
        )

        indices = self.subsample(range(len(dataset)))
        for idx in indices:
            row = dataset[int(idx)]

            # `num_turns` is authoritative where the release provides it;
            # counting user messages is the fallback.
            num_turns = row.get("num_turns")
            conversation_a = turns(row.get("conversation_a"))
            conversation_b = turns(row.get("conversation_b"))
            if num_turns is None:
                num_turns = max(count_exchanges(conversation_a),
                                count_exchanges(conversation_b))
            if num_turns != 1:
                # Kept out of the main protocol; the count is the answer to
                # "multi-turn items" in the bottleneck list.
                self.drop("multi-turn battle")
                continue

            instruction = first_user_text(conversation_a) or first_user_text(conversation_b)
            response_a = first_assistant_text(conversation_a)
            response_b = first_assistant_text(conversation_b)
            if not instruction or not response_a or not response_b:
                self.drop("empty instruction or response")
                continue

            winner = str(row.get("winner") or "").strip()
            is_tie = winner in TIE_VALUES
            label = WINNER_TO_LABEL.get(winner)
            if label is None and not is_tie:
                self.drop(f"unrecognised winner value {winner!r}")
                continue

            image = self._image(row)
            if image is None:
                self.drop("no image on row")
                continue
            ref = self.store.put(image, key=f"visionarena:{idx}")
            if ref is None:
                self.drop("image decode failed")
                continue

            annotator = str(row.get("judge") or row.get("user_id") or f"anon_{idx}")
            annotations = [Annotation(annotator=annotator,
                                      label="tie" if is_tie else label)]

            yield Record(
                uid=make_uid(self.name, row.get("question_id", idx), ref.sha256),
                source=self.name,
                group_id=make_group_id(self.name, [ref.sha256], instruction),
                modality="image",
                media=[ref.path],
                media_hash=[ref.phash],
                instruction=instruction,
                response_a=response_a,
                response_b=response_b,
                model_a=str(row.get("model_a") or "unknown"),
                model_b=str(row.get("model_b") or "unknown"),
                label=None if is_tie else label,
                label_origin="vote",
                annotations=[a.to_dict() for a in annotations],
                n_ann=1,
                agreement=None,          # one vote: agreement is undefined, not 1.0
                is_tie=is_tie,
                meta={
                    "row": int(idx),
                    "winner": winner,
                    "tie_type": winner if is_tie else None,
                    "annotator": annotator,
                    # A hash of the voting user: the one source where the same
                    # id really is the same person across instances.
                    "real_annotator_ids": True,
                    "lang": row.get("language"),
                    "category": active_categories(row.get("categories")),
                    "num_turns": row.get("num_turns"),
                    "len_a": len(response_a),
                    "len_b": len(response_b),
                    "img_res": [ref.width, ref.height],
                    "img_sha256": ref.sha256,
                },
            )
            self.keep()
