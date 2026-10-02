"""Stage-by-stage counters.

Assignment 2 asks how bias was tackled and what the bottleneck was. Both
answers are numbers, and numbers are only credible if they were recorded while
the rows were being dropped rather than reconstructed afterwards. Every stage
that removes or reclassifies a row calls `funnel.drop(...)`, and Phase 4 prints
the result as a funnel table.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class Funnel:
    def __init__(self) -> None:
        # stage -> source -> count of rows *surviving* that stage
        self.kept: Dict[str, Dict[str, int]] = OrderedDict()
        # stage -> source -> reason -> count of rows dropped
        self.dropped: Dict[str, Dict[str, Dict[str, int]]] = OrderedDict()
        self.notes: List[str] = []

    def _stage(self, stage: str) -> None:
        self.kept.setdefault(stage, defaultdict(int))
        self.dropped.setdefault(stage, defaultdict(lambda: defaultdict(int)))

    def keep(self, stage: str, source: str, n: int = 1) -> None:
        self._stage(stage)
        self.kept[stage][source] += n

    def drop(self, stage: str, source: str, reason: str, n: int = 1) -> None:
        self._stage(stage)
        self.dropped[stage][source][reason] += n

    def note(self, text: str) -> None:
        self.notes.append(text)

    # -- reporting --------------------------------------------------------

    def sources(self) -> List[str]:
        seen = []
        for stage in self.kept:
            for source in list(self.kept[stage]) + list(self.dropped.get(stage, {})):
                if source not in seen:
                    seen.append(source)
        return seen

    def to_dict(self) -> Dict:
        return {
            "kept": {stage: dict(counts) for stage, counts in self.kept.items()},
            "dropped": {
                stage: {source: dict(reasons) for source, reasons in by_source.items()}
                for stage, by_source in self.dropped.items()
            },
            "notes": self.notes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Funnel":
        funnel = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for stage, counts in data.get("kept", {}).items():
            for source, n in counts.items():
                funnel.keep(stage, source, n)
        for stage, by_source in data.get("dropped", {}).items():
            for source, reasons in by_source.items():
                for reason, n in reasons.items():
                    funnel.drop(stage, source, reason, n)
        funnel.notes = list(data.get("notes", []))
        return funnel

    def markdown_table(self, sources: Optional[List[str]] = None) -> str:
        """Rows = stages, columns = sources, cells = surviving count."""
        sources = sources or self.sources()
        if not sources:
            return "_No rows recorded._\n"

        header = "| Stage | " + " | ".join(sources) + " | Total |"
        rule = "|---" * (len(sources) + 2) + "|"
        lines = [header, rule]

        for stage, counts in self.kept.items():
            cells = [str(counts.get(source, 0)) for source in sources]
            total = sum(counts.get(source, 0) for source in sources)
            lines.append(f"| {stage} | " + " | ".join(cells) + f" | {total} |")

        return "\n".join(lines) + "\n"

    def markdown_drops(self, sources: Optional[List[str]] = None) -> str:
        sources = sources or self.sources()
        lines = ["| Stage | Source | Reason | Dropped |", "|---|---|---|---|"]
        any_row = False
        for stage, by_source in self.dropped.items():
            for source in sources:
                for reason, n in sorted(by_source.get(source, {}).items(),
                                        key=lambda kv: -kv[1]):
                    if n:
                        lines.append(f"| {stage} | {source} | {reason} | {n} |")
                        any_row = True
        if not any_row:
            return "_Nothing dropped._\n"
        return "\n".join(lines) + "\n"
