"""Track A inspection and individually authorized training/frozen evaluation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.benchmark_track_a import MODEL_CONFIGS, build_model, model_config, parameter_counts
from src.training.benchmark_track_a import train_run, evaluate_run, planned_matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("inspect", help="Count models and print the planned matrix; no training or files.")
    train = sub.add_parser("train", help="One explicitly approved 500-epoch run only.")
    train.add_argument("--model", choices=MODEL_CONFIGS, required=True)
    train.add_argument("--train-resolution", type=int, choices=(1200, 2399), required=True)
    train.add_argument("--resume", type=Path)
    evaluate = sub.add_parser("evaluate", help="One frozen checkpoint across the four registered Q400 grids.")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    for child in (train, evaluate):
        child.add_argument("--output", type=Path, required=True)
        child.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
        child.add_argument("--host-gpu", type=int, choices=range(4))
        child.add_argument("--execution-approved", action="store_true",
                           help="Acknowledge prior task-specific user approval; this flag does not grant approval.")
    args = parser.parse_args(argv)
    if args.action == "inspect":
        result = {"models": {name: {"config": model_config(name), **parameter_counts(build_model(name))}
                             for name in MODEL_CONFIGS}, "planned_matrix": planned_matrix()}
    elif args.action == "train":
        result = train_run(args.model, args.train_resolution, args.output, args.device,
                           args.host_gpu, args.execution_approved, args.resume)
    else:
        result = evaluate_run(args.checkpoint, args.output, args.device, args.host_gpu, args.execution_approved)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
