"""JudgeAnything: pairwise battles between any-to-any models.

The release is an any-to-any benchmark: 1,500 queries over 15 task types, of
which only **Image2Text** (100 queries) is an image-grounded comparison of two
*text* responses. The other 1,400 are either generation tasks, whose outputs
are images, video or audio and so cannot be compared by a text judge at all,
or understanding tasks whose input is video or audio and therefore out of
scope. See DATASET_FINDINGS.md section 7.

Assembling one record needs a three-way join, none of which is documented:

    X2XRawBenchmark/TaskAnything.json
        1,500 queries: uniq_id, task_name, question, image_path
    Arena/arenaresult/Result/gang_Understanding_record.json
        500 understanding uniq_ids -> battle outcome *strings*, 4 per query,
        of the form "<model_a> vs <model_b> <winner> wins"
    Arena/<model>/<Model>.json
        that model's answer for every uniq_id

The ids line up exactly across all three (500/500), but two traps cost time:

* `Arena/<model>/*.json` is **not** the source of responses for the
  ResponseCollection models. `ResponseCollection/JudgeAnything.json` holds
  answers from Phi3V, claude, gpt-4o and qwen-vl-max, which never appear in
  the battle records. The battles are between the eight *Arena* models, and
  their answers live in the per-model files.
* `answer` is a dict on some models and a list of dicts on others, and its
  `type` is spelled both "text" and "Text".

What this source does **not** provide is annotator ids: the battles are
aggregated outcomes, so it cannot feed Simulated Annotators (Ind.) or the
inter-annotator agreement analysis.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from prepare_data.adapters.base import Adapter
from prepare_data.schema import Annotation, Record, make_group_id, make_uid

TASK = "Image2Text"

# Battle strings name models by display name; answers live in these files.
MODEL_FILES = {
    "Baichuan-Omni-1.5": "Arena/bc-omni/Baichuan-Omni-1.5.json",
    "CoDi": "Arena/codi/Codi.json",
    "Gemini-1.5": "Arena/gemini/Gemini.json",
    "ModaVerse": "Arena/modaverse/ModaVerse.json",
    "Next-GPT": "Arena/nextgpt/nextgpt_output.json",
    "OneLLM": "Arena/onellm/OneLLM.json",
    "Unified-IO2": "Arena/unifiedio2/UnifiedIO2.json",
    "VideoLlama2": "Arena/videollama2/VideoLlama2.json",
}

TASKS_FILE = "X2XRawBenchmark/TaskAnything.json"
BATTLES_FILE = "Arena/arenaresult/Result/gang_Understanding_record.json"
CHECKLIST_FILE = "Checklist/Image2Text.json"
IMAGE_ROOT = "X2XRawBenchmark"

# Outcomes that mean "no winner" rather than naming one.
TIE_TOKENS = {"draw", "tie"}


def image_paths(value: Any) -> List[str]:
    """`image_path` is sometimes a list, sometimes its repr as a string."""
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value or "").strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            return [str(v) for v in parsed] if isinstance(parsed, list) else [text]
        except (ValueError, SyntaxError):
            return [text]
    return [text] if text else []


def answer_text(answer: Any) -> str:
    """Flatten the several shapes `answer` takes into plain text.

    Seen in the wild: {"type": "text"|"Text", "content": str | [str]}, and a
    list of such dicts (ModaVerse, Next-GPT).
    """
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer.strip()
    if isinstance(answer, list):
        return "\n".join(filter(None, (answer_text(item) for item in answer))).strip()
    if isinstance(answer, dict):
        if str(answer.get("type", "")).lower() not in ("text", ""):
            return ""          # a generated media answer: nothing to compare
        content = answer.get("content")
        if isinstance(content, list):
            return "\n".join(str(c) for c in content if c).strip()
        return str(content or "").strip()
    return ""


def parse_battle(text: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Parse "<model_a> vs <model_b> <winner> wins" into (a, b, winner).

    Returns winner=None for a draw or an outcome with no winner named. Model
    names are matched against MODEL_FILES rather than split on whitespace,
    because several contain hyphens and dots and the trailing winner token is
    not delimited.
    """
    if " vs " not in text:
        return None
    left, right = text.split(" vs ", 1)
    model_a = left.strip()
    if model_a not in MODEL_FILES:
        return None

    rest = right.strip()
    for model_b in sorted(MODEL_FILES, key=len, reverse=True):
        if rest == model_b:
            return model_a, model_b, None                 # no outcome recorded
        if not rest.startswith(model_b):
            continue
        tail = rest[len(model_b):].strip()
        if not tail or tail.lower() in TIE_TOKENS:
            return model_a, model_b, None
        if tail.endswith(" wins"):
            winner = tail[: -len(" wins")].strip()
            if winner in MODEL_FILES:
                return model_a, model_b, winner
            return model_a, model_b, None
        return model_a, model_b, None
    return None


class JudgeAnythingAdapter(Adapter):
    name = "judge_anything"

    def __init__(self, *args, modalities=("image",), **kwargs):
        super().__init__(*args, **kwargs)
        self.modalities = tuple(modalities)

    # -- loading ----------------------------------------------------------

    def _json(self, relative: str) -> Any:
        path = self.hf_dir() / relative
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _answers(self) -> Dict[str, Dict[str, str]]:
        """{model name: {uniq_id: answer text}} for the eight Arena models."""
        answers: Dict[str, Dict[str, str]] = {}
        for model, relative in MODEL_FILES.items():
            rows = self._json(relative)
            if not isinstance(rows, list):
                self.funnel.note(f"judge_anything: {relative} missing or unreadable.")
                continue
            answers[model] = {
                row["uniq_id"]: answer_text(row.get("answer"))
                for row in rows
                if isinstance(row, dict) and row.get("uniq_id")
            }
        return answers

    def _checklists(self) -> Dict[str, List[str]]:
        rows = self._json(CHECKLIST_FILE)
        if not isinstance(rows, list):
            return {}
        out: Dict[str, List[str]] = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get("uniq_id"):
                continue
            items = [c.get("checklist") for c in (row.get("checklists") or [])
                     if isinstance(c, dict) and c.get("checklist")]
            out.setdefault(row["uniq_id"], []).extend(items)
        return out

    # -- main --------------------------------------------------------------

    def iter_records(self) -> Iterator[Record]:
        if "image" not in self.modalities:
            self.funnel.note("judge_anything: skipped, image not in --modalities.")
            return

        tasks = self._json(TASKS_FILE)
        battles = self._json(BATTLES_FILE)
        if not isinstance(tasks, list) or not isinstance(battles, dict):
            self.funnel.note(
                f"judge_anything: {TASKS_FILE} or {BATTLES_FILE} missing; adapter "
                "skipped. Run `python -m prepare_data.download --source judge_anything`.")
            return

        by_uid = {t["uniq_id"]: t for t in tasks if isinstance(t, dict) and t.get("uniq_id")}
        answers = self._answers()
        checklists = self._checklists()

        image_task_ids = [uid for uid in battles
                          if by_uid.get(uid, {}).get("task_name") == TASK]
        self.funnel.note(
            f"judge_anything: {len(tasks)} queries over 15 task types; {TASK} is "
            f"{len(image_task_ids)} of them. The other task types are generation "
            "(output is media, nothing for a text judge to compare) or have video "
            "or audio input."
        )
        self.funnel.note(
            "judge_anything: battle outcomes are aggregated, with no annotator ids, "
            "so this source cannot feed Ind. pools or inter-annotator agreement."
        )

        for uid in self.subsample(sorted(image_task_ids)):
            task = by_uid[uid]

            media = image_paths(task.get("image_path"))
            if len(media) != 1:
                # The judge prompt carries one image; 23 of the 100 queries are
                # multi-image story tasks.
                self.drop(f"multi-image task ({len(media)} images)", len(battles[uid]))
                continue

            instruction = str(task.get("question") or "").strip()
            if not instruction:
                self.drop("empty question", len(battles[uid]))
                continue

            image_file = self.hf_dir() / IMAGE_ROOT / media[0]
            ref = self.store.put_path(image_file)
            if ref is None:
                self.drop("image missing or undecodable", len(battles[uid]))
                continue

            for outcome in battles[uid]:
                parsed = parse_battle(str(outcome))
                if parsed is None:
                    self.drop("unparsable battle string")
                    continue
                model_a, model_b, winner = parsed

                response_a = answers.get(model_a, {}).get(uid, "")
                response_b = answers.get(model_b, {}).get(uid, "")
                if not response_a or not response_b:
                    self.drop("missing response for one side")
                    continue

                is_tie = winner is None
                label = None if is_tie else ("A" if winner == model_a else "B")

                # One aggregated outcome, so a single placeholder annotator.
                annotations = [Annotation(annotator="judge_anything_panel",
                                          label="tie" if is_tie else label)]

                yield Record(
                    uid=make_uid(self.name, uid, model_a, model_b),
                    source=self.name,
                    group_id=make_group_id(self.name, [ref.sha256], instruction),
                    modality="image",
                    media=[ref.path],
                    media_hash=[ref.phash],
                    instruction=instruction,
                    response_a=response_a,
                    response_b=response_b,
                    model_a=model_a,
                    model_b=model_b,
                    label=label,
                    label_origin="vote",
                    annotations=[a.to_dict() for a in annotations],
                    n_ann=1,
                    agreement=None,       # aggregated outcome, not per-annotator
                    is_tie=is_tie,
                    meta={
                        "uniq_id": uid,
                        "task": TASK,
                        "tie_type": "no winner recorded" if is_tie else None,
                        "real_annotator_ids": False,
                        # Kept for analysis, deliberately out of the judge prompt.
                        "checklist": checklists.get(uid),
                        "len_a": len(response_a),
                        "len_b": len(response_b),
                        "img_res": [ref.width, ref.height],
                        "img_sha256": ref.sha256,
                    },
                )
                self.keep()
