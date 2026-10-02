"""Paths, source registry and shared constants for the multimodal preference pipeline.

Output of this package is `data/vlm_v2/`, a drop-in replacement for the
`data/vlm/` tree written by the old `prepare_vlm_dataset.py`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(os.environ.get("VLM_DATA_ROOT", REPO_ROOT / "data"))
RAW_ROOT = DATA_ROOT / "raw"           # downloads + manual drops, one dir per source
OUT_ROOT = DATA_ROOT / "vlm_v2"        # canonical records
MEDIA_ROOT = OUT_ROOT / "media"        # content-addressed image store
EXPORT_ROOT = DATA_ROOT / "vlm_v2_export"   # repo-format splits for the judge scripts
REPORT_ROOT = REPO_ROOT / "reports"

# The HF cache gets large (MM-RLHF and VisionArena images). On Delta, export
# HF_HOME=/scratch/... before running; we never silently redirect it.
HF_HOME = os.environ.get("HF_HOME")

# --------------------------------------------------------------------------
# Pipeline constants
# --------------------------------------------------------------------------

SEED = 42

# Simulated Annotators grid. A VLM prompt embeds one image per few-shot
# example, so K stays far below the K <= 5 used in the text-only paper.
K_VALUES = (1, 2)
N_VALUES = (1, 2, 3, 5)
K_MAX = max(K_VALUES)
N_MAX = max(N_VALUES)

# Split fractions over *groups*, not rows, so near-duplicate instances cannot
# straddle a split boundary.
SPLIT_FRACTIONS = {"fewshot": 0.10, "dev": 0.10, "eval_pool": 0.80}

# Share of the fewshot split reserved for annotator-dense records, so that
# Simulated Annotators (Ind.) has annotators clearing K+1 inside the split.
# Capped well below 1.0 because only VisionArena carries annotator ids: filling
# the whole quota that way leaves the Maj. pools demonstrating one dataset's
# style to a judge that has to grade four.
FEWSHOT_DENSITY_SHARE = 0.5

# MM-RLHF: minimum gap in the ranking score before a pair counts as a clear
# preference. Fixed on dev, swept for the bias table.
MM_RLHF_TAU = 1.0
MM_RLHF_TAU_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0)

# Judge-time context budget. Used only to *flag* oversized pairs; nothing is
# dropped for this reason without it appearing in the funnel table.
JUDGE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
CONTEXT_LIMIT_TOKENS = 32768
IMAGE_TOKEN_ESTIMATE = 1280   # ~tokens for one image at the judge's max_pixels

# Cross-source dedup priority: on a collision the record from the source
# further down this list is dropped. Richer annotation metadata wins.
SOURCE_PRIORITY = [
    "visit_bench",      # 5 judgments/item -> majority + agreement
    "visionarena",      # 1 vote/item, but real user ids -> the Ind. pools
    "judge_anything",   # aggregated battle outcomes, no per-annotator labels
    "mm_rlhf",          # ranking-derived
    "rlhf_v",           # correction-derived, pilot only
]

# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------

SOURCES = {
    "rlhf_v": {
        "kind": "hf",
        "repo": "openbmb/RLHF-V-Dataset",
        "split": "train",
        "gated": False,
        "modality": "image",
        "home": "https://github.com/RLHF-V/RLHF-V",
        "note": ("Correctional feedback. The preferred side is always the human edit, "
                 "so this is a pilot source, not a main result."),
    },
    "visit_bench": {
        "kind": "url",
        # The pairwise human judgments live in the GitHub repo, not the HF mirror
        # (the HF `mlfoundations/VisIT-Bench` config is the 574-row single-model set).
        "url": ("https://raw.githubusercontent.com/mlfoundations/VisIT-Bench/main/"
                "visit_bench_human_preferences.csv"),
        "filename": "visit_bench_human_preferences.csv",
        "gated": False,
        "modality": "image",
        "home": "https://github.com/mlfoundations/VisIT-Bench",
        "note": "Images referenced by URL and fetched separately; 5 judgments per tuple.",
    },
    "judge_anything": {
        "kind": "hf",
        "repo": "pudashi/JudgeAnything",
        "split": None,       # resolved at load time
        "gated": False,
        "modality": "image",
        "home": "https://urrealhero.github.io/judgeanythingweb/",
        # The repo is ~10,300 individual files covering all 15 modality
        # combinations, and fetching it whole runs to hours. The image protocol
        # needs the JSON annotations and image media only; .mp4 and .wav are
        # the video and audio sets. Add those patterns here when the extension
        # sets are taken up.
        "allow_patterns": ["*.json", "*.jsonl", "*.csv", "*.md",
                           "*.png", "*.jpg", "*.jpeg", "*.webp"],
        # An any-to-any benchmark: 1,500 queries over 15 task types, of which
        # only Image2Text (100) is an image-grounded comparison of two text
        # responses. 77 are single-image, giving ~307 pairs. Records are built
        # by joining TaskAnything.json, gang_Understanding_record.json and the
        # per-model Arena files; see the adapter docstring for the two traps.
        # Battle outcomes are aggregated, so there are no annotator ids - the
        # "12 annotators, 15 per item" in the Assignment 1 report does not
        # match what shipped.
        "note": ("Image2Text only (~307 pairs). Aggregated battle outcomes, so no "
                 "annotator ids; cannot feed the Ind. pools."),
    },
    "mm_rlhf": {
        "kind": "hf",
        "repo": "yifanzhang114/MM-RLHF",
        "split": "train",
        "gated": False,
        "modality": "image",
        "home": "https://huggingface.co/datasets/yifanzhang114/MM-RLHF",
        # The metadata parquet is ~35 MB, but the `image` column is a relative
        # path into zip archives shipped in the same repo, so a usable download
        # is tens of GB. `video.zip` is pure video: its 2,855 rows carry no
        # image at all and the image protocol drops them, so it is never fetched.
        # The image archives are short/ (6,447 rows), long/ (4,704),
        # mcq/ (1,352) and safety/ (984, excluded at the record level).
        "ignore_patterns": ["video.zip"],
        "note": ("Metadata plus multi-GB image archives in the same repo; video.zip "
                 "is excluded at download time and safety/ at the record level."),
    },
    "visionarena": {
        "kind": "hf",
        "repo": "lmarena-ai/VisionArena-Battle",
        "split": "train",
        "gated": True,       # gated:auto -> accept the terms, then log in
        "modality": "image",
        "home": "https://huggingface.co/datasets/lmarena-ai/VisionArena-Battle",
        "note": ("Gated. Accept the terms on the dataset page while signed in, then run "
                 "`hf auth login` (or `huggingface-cli login`) before downloading."),
    },
}

ALL_SOURCES = list(SOURCES)


def raw_dir(source: str) -> Path:
    """Where a source's raw download (or manual drop) lives."""
    return RAW_ROOT / source


def ensure_dirs() -> None:
    for path in (DATA_ROOT, RAW_ROOT, OUT_ROOT, MEDIA_ROOT, EXPORT_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def use_utf8_stdout() -> None:
    """Stop Windows consoles from crashing on non-ASCII content.

    VisionArena spans 90 languages, so any diagnostic that echoes an
    instruction can hit a character the default cp1252 console cannot encode
    and kill the run. Replacing unencodable characters is the right trade here:
    a garbled character in a progress line is better than a lost pipeline run,
    and the data files themselves are always written as UTF-8 regardless.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
