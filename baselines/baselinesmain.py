import argparse
import json
from pathlib import Path

from baselines.steering2.config2 import ATS_PATH


def _cmd_train(args: argparse.Namespace) -> None:
    from baselines.calibration2.train_ats2 import train

    if Path(ATS_PATH).exists() and not args.overwrite:
        raise SystemExit(
            f"ATS weights already exist at {ATS_PATH}. Use --overwrite to retrain."
        )

    train(split=args.split)
    print(f"Saved ATS head to {ATS_PATH}")


def _cmd_eval(args: argparse.Namespace) -> None:
    from baselines.baseline_ats import evaluate_baseline

    metrics, csv_path = evaluate_baseline(split=args.split, csv_name=args.csv_name)
    print(f"Saved per-sample probabilities to {csv_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baselines runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-ats", help="Fit the ATS calibration head")
    train_parser.add_argument("--split", default="validation", help="MedQA split for calibration")
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing checkpoint at ATS_PATH",
    )
    train_parser.set_defaults(func=_cmd_train)

    eval_parser = subparsers.add_parser("evaluate", help="Run baseline evaluation")
    eval_parser.add_argument("--split", default="validation", help="MedQA split for evaluation")
    eval_parser.add_argument(
        "--csv-name",
        default="baseline_ats.csv",
        help="Filename for saving per-sample outputs",
    )
    eval_parser.set_defaults(func=_cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()