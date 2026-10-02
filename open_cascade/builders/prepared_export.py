"""Use the corpus produced by get_data.py without rebuilding or resplitting it."""
from typing import Dict, List

from open_cascade.builders.base import DatasetBuilder
from open_cascade.data import read_jsonl


class PreparedExportBuilder(DatasetBuilder):
    def load_instances(self) -> List[Dict]:
        return read_jsonl(self.config.split_file("eval_pool"))

    def build(self) -> None:
        if not self.config.flat_files:
            raise ValueError("prepared_export requires data.flat_files=true")
        pool = self.config.split_file("eval_pool")
        fewshot = self.config.fewshot_file
        read_jsonl(pool)
        read_jsonl(fewshot)
        print(f"Prepared export ready: {pool} and {fewshot}")
