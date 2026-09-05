"""Calibrate and evaluate a cascade of open judges.

Because judge scoring is cached per model in ./result/, any cascade subset or ordering can
be evaluated post-hoc with no further inference -- this is the whole of the paper's judge
composition ablation (Table 6).

Usage:
    python run_open_cascade.py --alpha=0.15
    python run_open_cascade.py --model_names mistral-7b-instruct qwen2.5-7b-instruct
    python run_open_cascade.py --alpha_sweep
"""
from argparse import ArgumentParser

from open_cascade.cascade import OpenCascadedClassifier
from open_cascade.data import load_judgements
from open_cascade.registry import DEFAULT_OPEN_CASCADE


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_names", nargs="+", default=DEFAULT_OPEN_CASCADE,
                        help="Judges, weakest first")
    parser.add_argument("--result_dir", default="./result")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="Risk tolerance; target human agreement is 1 - alpha")
    parser.add_argument("--delta", type=float, default=0.1, help="Error level")
    parser.add_argument("--alpha_sweep", action="store_true",
                        help="Sweep alpha instead of evaluating a single value")
    parser.add_argument("--no_split_delta", action="store_true",
                        help="Calibrate every judge at the full delta, reproducing the "
                             "released code instead of Algorithm 2")
    return parser.parse_args()


def evaluate(model_names, calibration, test, alpha, delta, split_delta):
    classifier = OpenCascadedClassifier(
        model_names, calibration_samples=calibration,
        alpha=alpha, delta=delta, split_delta=split_delta)
    composition, selective_acc, coverage, _ = classifier.apply_decision_rule(test)
    return classifier, composition, selective_acc, coverage


if __name__ == "__main__":
    args = parse_args()
    split_delta = not args.no_split_delta

    calibration = load_judgements(args.model_names, "calibration", args.result_dir)
    test = load_judgements(args.model_names, "test", args.result_dir)

    print(f"Cascade: {' -> '.join(args.model_names)}")
    print(f"delta={args.delta}"
          f"{f' (split as {args.delta}/{len(args.model_names)} per judge)' if split_delta else ' (not split)'}\n")

    if args.alpha_sweep:
        alphas = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
        header = f"{'target':>7} {'agreement':>10} {'coverage':>9}  composition"
        print(header)
        print("-" * (len(header) + 20))
        for alpha in alphas:
            _, composition, acc, coverage = evaluate(
                args.model_names, calibration, test, alpha, args.delta, split_delta)
            shares = "  ".join(f"{n}:{composition[n]:.0%}" for n in args.model_names)
            flag = "" if acc >= 1 - alpha else "  <- MISSED"
            print(f"{1 - alpha:>7.2f} {acc:>10.4f} {coverage:>9.3f}  {shares}{flag}")
        print("\nNote: this is a single split. The paper's headline metric is the guarantee")
        print("success rate over 1000 random calibration/test splits.")
    else:
        classifier, composition, acc, coverage = evaluate(
            args.model_names, calibration, test, args.alpha, args.delta, split_delta)
        print(f"lambda_hats           : {[round(float(l), 4) for l in classifier.lambda_hats]}")
        print(f"target human agreement: {1 - args.alpha:.2f}")
        print(f"empirical agreement   : {acc:.4f}")
        print(f"coverage              : {coverage:.4f}")
        print(f"evaluator composition : "
              f"{ {name: round(share, 4) for name, share in composition.items()} }")
