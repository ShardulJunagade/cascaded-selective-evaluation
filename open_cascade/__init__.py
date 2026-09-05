"""Open-model replication of Cascaded Selective Evaluation.

The paper's cascade is Mistral-7B-Instruct-v0.2 -> gpt-3.5-turbo -> gpt-4-turbo. This
package replaces the two OpenAI judges with open-weight models so the experiment can be run
without API access.

Nothing under `cascaded_evaluation/` or `model/` is modified. The statistical core --
`SelectiveClassificationUtil`, `merge_data`, `prepare_data` -- is imported from the authors'
`cascaded_evaluation.util` and reused as-is; only judge inference and the cascade loop are
reimplemented here.

Layout:
    registry.py  judge configurations (HF name, dtype, tensor parallelism, ...)
    judge.py     teacher-forced vLLM scoring for Simulated Annotators
    cascade.py   cascade calibration + decision rule
    data.py      jsonl / split helpers
"""
