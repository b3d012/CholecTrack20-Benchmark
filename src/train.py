"""Command line entrypoint for training, tracking, ablations, and checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .strongsort_tracker import TrackingConfig, benchmark_trackers, run_tracker
from .utils import append_csv_row, check_environment, save_json, validate_yolo_dataset
from .yolov11_model import DetectorConfig, train_yolov11, validate_yolov11


def cmd_check_env(_: argparse.Namespace) -> None:
    info = check_environment()
    save_json(info, "results/logs/environment_check.json")
    for key, value in info.items():
        print(f"{key}: {value}")


def cmd_validate_data(args: argparse.Namespace) -> None:
    problems = validate_yolo_dataset(args.data)
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)
    print(f"Dataset YAML is valid: {args.data}")


def cmd_train_detector(args: argparse.Namespace) -> None:
    config = DetectorConfig(
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    train_yolov11(config)


def cmd_validate_detector(args: argparse.Namespace) -> None:
    validate_yolov11(args.weights, args.data, args.device)


def cmd_track(args: argparse.Namespace) -> None:
    run_tracker(
        TrackingConfig(
            weights=args.weights,
            source=args.source,
            tracker=args.tracker,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            name=f"tracking_{args.tracker}",
        )
    )


def cmd_benchmark_trackers(args: argparse.Namespace) -> None:
    benchmark_trackers(args.weights, args.source, args.trackers, args.device)


def cmd_ablate(args: argparse.Namespace) -> None:
    experiments = [
        {"name": "baseline", "model": args.model, "giou": False, "efficientnetv2": False, "eca": False},
        {"name": "giou", "model": args.model, "giou": True, "efficientnetv2": False, "eca": False},
        {"name": "efficientnetv2", "model": args.model, "giou": False, "efficientnetv2": True, "eca": False},
        {"name": "eca", "model": args.model, "giou": False, "efficientnetv2": False, "eca": True},
        {"name": "efficientnetv2_eca", "model": args.model, "giou": False, "efficientnetv2": True, "eca": True},
    ]

    log_path = Path("results/logs/ablation_plan.csv")
    for experiment in experiments:
        append_csv_row(log_path, experiment)
        if args.dry_run:
            print(f"Planned ablation: {experiment['name']}")
            continue
        train_yolov11(
            DetectorConfig(
                model=experiment["model"],
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                name=f"ablation_{experiment['name']}",
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EA-StrongSORT project CLI")
    subparsers = parser.add_subparsers(required=True)

    check_env = subparsers.add_parser("check-env")
    check_env.set_defaults(func=cmd_check_env)

    validate_data = subparsers.add_parser("validate-data")
    validate_data.add_argument("--data", required=True)
    validate_data.set_defaults(func=cmd_validate_data)

    train = subparsers.add_parser("train-detector")
    train.add_argument("--data", required=True)
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="0")
    train.add_argument("--project", default="results/logs")
    train.add_argument("--name", default="yolov11_cholectrack20")
    train.set_defaults(func=cmd_train_detector)

    val = subparsers.add_parser("validate-detector")
    val.add_argument("--weights", required=True)
    val.add_argument("--data", required=True)
    val.add_argument("--device", default="0")
    val.set_defaults(func=cmd_validate_detector)

    track = subparsers.add_parser("track")
    track.add_argument("--weights", required=True)
    track.add_argument("--source", required=True)
    track.add_argument("--tracker", default="strongsort")
    track.add_argument("--device", default="0")
    track.add_argument("--conf", type=float, default=0.25)
    track.add_argument("--iou", type=float, default=0.7)
    track.set_defaults(func=cmd_track)

    benchmark = subparsers.add_parser("benchmark-trackers")
    benchmark.add_argument("--weights", required=True)
    benchmark.add_argument("--source", required=True)
    benchmark.add_argument("--trackers", nargs="+", default=["botsort", "bytetrack", "ocsort", "strongsort"])
    benchmark.add_argument("--device", default="0")
    benchmark.set_defaults(func=cmd_benchmark_trackers)

    ablate = subparsers.add_parser("ablate")
    ablate.add_argument("--data", required=True)
    ablate.add_argument("--model", default="yolo11n.pt")
    ablate.add_argument("--epochs", type=int, default=100)
    ablate.add_argument("--imgsz", type=int, default=640)
    ablate.add_argument("--batch", type=int, default=16)
    ablate.add_argument("--device", default="0")
    ablate.add_argument("--dry-run", action="store_true")
    ablate.set_defaults(func=cmd_ablate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
