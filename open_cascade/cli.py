"""Command-line entry point.

    python -m open_cascade <command> <config.yaml> [--set key=value ...]

Commands (normally run in this order):

    prepare-data   build the dataset splits described by the config's `data` section
    score          run each judge over the splits (GPU, vLLM) and cache its judgements
    evaluate       calibrate the cascade on cached judgements and run the experiments
    validate       compare a re-scored judge against released judgements
    list           show the registered judges, dataset builders, experiments and
                   threshold methods (takes no config)

`--set` overrides any config value without editing the file, e.g.

    python -m open_cascade evaluate configs/text_mistral_qwen7b.yaml --set cascade.alpha=0.1
"""
import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from open_cascade.config import ExperimentConfig, load_config


def cmd_prepare_data(config: ExperimentConfig) -> None:
    from open_cascade.builders import get_builder

    print(f"Building '{config.data.builder}' dataset into {config.data.out_dir}")
    get_builder(config.data).build()


def cmd_score(config: ExperimentConfig, args) -> None:
    from open_cascade.scoring import score_judge

    if args.judge is not None:
        score_judge(config, args.judge)
        return
    if len(config.judges) == 1:
        score_judge(config, config.judges[0])
        return

    # One process per judge: vLLM does not reliably free GPU memory when a model is
    # unloaded, so loading the next judge in the same process can run out of memory.
    for judge in config.judges:
        print(f"\n=== scoring {judge} ===")
        command = [sys.executable, "-m", "open_cascade", "score", args.config,
                   "--judge", judge]
        for override in args.set:
            command += ["--set", override]
        subprocess.run(command, check=True)


def cmd_evaluate(config: ExperimentConfig) -> None:
    from open_cascade.data import load_judgements
    from open_cascade.experiments import EXPERIMENTS

    unknown = [name for name in config.evaluate.experiments if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s) {unknown}. Known: {sorted(EXPERIMENTS)}")

    calibration = load_judgements(config.judges, "calibration", config.result_dir)
    test = load_judgements(config.judges, "test", config.result_dir)

    cascade = config.cascade
    delta_note = (f"split as {cascade.delta}/{len(config.judges)} per judge"
                  if cascade.split_delta else "not split")
    print(f"Cascade: {' -> '.join(config.judges)}")
    print(f"delta={cascade.delta} ({delta_note}), "
          f"threshold method: {cascade.threshold_method}")

    out_dir = Path(config.evaluate.output_dir) / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")

    for name in config.evaluate.experiments:
        print(f"\n=== {name} ===")
        result = EXPERIMENTS[name](config, calibration, test)
        out_file = out_dir / f"{name}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nsaved -> {out_file}")


def cmd_validate(config: ExperimentConfig) -> None:
    from open_cascade.validation import validate

    sys.exit(0 if validate(config) else 1)


def cmd_list() -> None:
    from open_cascade.builders import BUILDERS
    from open_cascade.experiments import EXPERIMENTS
    from open_cascade.registry import JUDGE_REGISTRY
    from open_cascade.thresholds import THRESHOLD_METHODS

    print("Judges (open_cascade/registry.py):")
    for name, judge in JUDGE_REGISTRY.items():
        print(f"  {name:<26} {judge.modality:<5} {judge.hf_name}  "
              f"(GPUs: {judge.tensor_parallel_size})")
    print("\nDataset builders (open_cascade/builders/):")
    for name, cls in BUILDERS.items():
        print(f"  {name:<26} {cls.__name__}")
    print("\nExperiments (open_cascade/experiments/):")
    for name, fn in EXPERIMENTS.items():
        print(f"  {name:<26} {(fn.__doc__ or '').strip().splitlines()[0]}")
    print("\nThreshold methods (open_cascade/thresholds.py):")
    for name in THRESHOLD_METHODS:
        print(f"  {name}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m open_cascade",
                            description="Cascaded Selective Evaluation pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str, help_text: str) -> ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", help="Experiment config, e.g. configs/vlm_qwen_3b_7b.yaml")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                             help="Override a config value, e.g. cascade.alpha=0.1 (repeatable)")
        return command

    add_command("prepare-data", "Build the dataset splits")
    score = add_command("score", "Score judges over the splits (GPU)")
    score.add_argument("--judge", default=None,
                       help="Score only this judge (default: every judge in the config)")
    add_command("evaluate", "Calibrate the cascade and run the experiments")
    add_command("validate", "Compare a re-scored judge against released judgements")
    commands.add_parser("list", help="Show registered judges, builders, experiments")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        cmd_list()
        return

    config = load_config(args.config, args.set)
    if args.command == "prepare-data":
        cmd_prepare_data(config)
    elif args.command == "score":
        cmd_score(config, args)
    elif args.command == "evaluate":
        cmd_evaluate(config)
    elif args.command == "validate":
        cmd_validate(config)
