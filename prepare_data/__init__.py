"""Multimodal preference dataset pipeline for VLM cascaded selective evaluation.

Phases (see reports/ for the artefacts each one writes):
    download    fetch raw sources, log size/time/failures
    inspect     raw-level statistics before any transformation
    adapt       per-source conversion to the canonical record schema
    stages      hygiene, dedup, tie routing, position randomization, group split
    stats       funnel + bias tables
    export      repo-format splits for run_open_vlm_judge.py
"""

__all__ = ["config", "schema", "media", "download", "adapters", "stages", "stats", "export"]
