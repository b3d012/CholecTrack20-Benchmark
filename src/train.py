"""Command line entrypoint for training, tracking, ablations, and checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from .cholectrack20 import (
    PERSPECTIVE_FIELDS,
    convert_dataset,
    export_mot_ground_truth,
    validate_yolo_labels,
)
from .metrics import detector_metrics_row, write_detection_metrics, write_tracking_metrics
from .strongsort_tracker import TrackingConfig, benchmark_trackers, run_tracker
from .utils import append_csv_row, check_environment, load_yaml, save_json, validate_yolo_dataset
from .yolov11_model import DetectorConfig, train_yolov11, validate_yolov11

SMOKE_BENCHMARK = {
    "videos": ["VID01"],
    "perspectives": ["visibility"],
    "trackers": ["botsort", "bytetrack"],
    "vid_stride": 25,
    "jobs": 2,
}


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


def _detector_config_from_args(args: argparse.Namespace) -> DetectorConfig:
    data = load_yaml(args.config) if args.config else {}
    return DetectorConfig(
        model=args.model or data.get("model", "yolo11n.pt"),
        data=args.data or data.get("data", "dataset/yolo_cholecTrack20/cholecTrack20.yaml"),
        epochs=args.epochs if args.epochs is not None else int(data.get("epochs", 100)),
        imgsz=args.imgsz if args.imgsz is not None else int(data.get("imgsz", 640)),
        batch=args.batch if args.batch is not None else int(data.get("batch", 16)),
        device=args.device or str(data.get("device", "0")),
        project=args.project or data.get("project", "results/logs"),
        name=args.name or data.get("name", "yolov11_cholectrack20"),
        lr0=args.lr0 if args.lr0 is not None else float(data.get("lr0", 0.01)),
        lrf=args.lrf if args.lrf is not None else float(data.get("lrf", 0.01)),
        optimizer=args.optimizer or data.get("optimizer", "SGD"),
        weight_decay=args.weight_decay
        if args.weight_decay is not None
        else float(data.get("weight_decay", 0.0005)),
        warmup_epochs=args.warmup_epochs
        if args.warmup_epochs is not None
        else int(data.get("warmup_epochs", 3)),
        weights_dir=data.get("weights_dir", "results/weights"),
    )


def cmd_prepare_data(args: argparse.Namespace) -> None:
    summaries = convert_dataset(args.source, args.out)
    problems = validate_yolo_labels(args.out)
    save_json(
        {"summaries": [summary.__dict__ for summary in summaries], "problems": problems},
        "results/logs/dataset_conversion_summary.json",
    )
    for summary in summaries:
        print(
            f"{summary.split}: {summary.videos} videos, {summary.images} images, "
            f"{summary.labels} labels, {summary.objects} objects, "
            f"{summary.skipped_frames} skipped missing frames"
        )
    if problems:
        for problem in problems[:25]:
            print(f"ERROR: {problem}")
        raise SystemExit(1)
    print(f"Dataset YAML written to {Path(args.out) / 'cholecTrack20.yaml'}")


def cmd_train_detector(args: argparse.Namespace) -> None:
    train_yolov11(_detector_config_from_args(args))


def cmd_validate_detector(args: argparse.Namespace) -> None:
    result = validate_yolov11(args.weights, args.data, args.device, args.split)
    row = detector_metrics_row(result, args.split)
    write_detection_metrics("results/logs/detection_metrics.csv", row)
    for key, value in row.items():
        print(f"{key}: {value}")


def cmd_evaluate_detector(args: argparse.Namespace) -> None:
    cmd_validate_detector(args)


def cmd_track(args: argparse.Namespace) -> None:
    run_tracker(
        TrackingConfig(
            weights=args.weights,
            source=args.source,
            tracker=args.tracker,
            perspective=args.perspective,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            name=f"tracking_{args.tracker}",
            save=args.save,
            vid_stride=args.vid_stride,
        )
    )


def cmd_benchmark_trackers(args: argparse.Namespace) -> None:
    if args.preset == "smoke":
        if not args.videos:
            args.videos = SMOKE_BENCHMARK["videos"]
        if not args.perspectives:
            args.perspectives = SMOKE_BENCHMARK["perspectives"]
        if not args.trackers:
            args.trackers = SMOKE_BENCHMARK["trackers"]
        if args.vid_stride is None:
            args.vid_stride = SMOKE_BENCHMARK["vid_stride"]
        if args.jobs is None:
            args.jobs = SMOKE_BENCHMARK["jobs"]

    args.perspectives = args.perspectives or ["visibility", "intracorporeal", "intraoperative"]
    args.trackers = args.trackers or ["botsort", "bytetrack", "ocsort", "strongsort"]
    args.vid_stride = args.vid_stride or 1
    args.jobs = args.jobs or 1

    invalid = [perspective for perspective in args.perspectives if perspective not in PERSPECTIVE_FIELDS]
    if invalid:
        raise SystemExit(f"Unsupported perspective(s): {', '.join(invalid)}")

    export_mot_ground_truth(
        args.gt_source,
        "results/logs/mot_ground_truth",
        args.perspectives,
        split_name=args.gt_split,
    )
    rows = benchmark_trackers(
        args.weights,
        args.source,
        args.trackers,
        args.perspectives,
        args.device,
        videos=args.videos,
        vid_stride=args.vid_stride,
        jobs=args.jobs,
    )
    write_tracking_metrics("results/logs/tracking_summary.csv", rows)
    for perspective in args.perspectives:
        perspective_rows = [row for row in rows if row["perspective"] == perspective]
        write_tracking_metrics(f"results/logs/tracking_{perspective}.csv", perspective_rows)
    for row in rows:
        print(
            f"{row['perspective']} {row['tracker']}: {row['status']} "
            f"({row['detections']} detections, {row['ids']} ids)"
        )


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

    prepare_data = subparsers.add_parser("prepare-data")
    prepare_data.add_argument("--source", default="dataset/cholecTrack20")
    prepare_data.add_argument("--out", default="dataset/yolo_cholecTrack20")
    prepare_data.set_defaults(func=cmd_prepare_data)

    train = subparsers.add_parser("train-detector")
    train.add_argument("--config", default="configs/yolov11_cholecTrack20.yaml")
    train.add_argument("--data")
    train.add_argument("--model")
    train.add_argument("--epochs", type=int)
    train.add_argument("--imgsz", type=int)
    train.add_argument("--batch", type=int)
    train.add_argument("--device")
    train.add_argument("--project")
    train.add_argument("--name")
    train.add_argument("--lr0", type=float)
    train.add_argument("--lrf", type=float)
    train.add_argument("--optimizer")
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--warmup-epochs", type=int)
    train.set_defaults(func=cmd_train_detector)

    val = subparsers.add_parser("validate-detector")
    val.add_argument("--weights", required=True)
    val.add_argument("--data", default="dataset/yolo_cholecTrack20/cholecTrack20.yaml")
    val.add_argument("--device", default="0")
    val.add_argument("--split", default="val", choices=["train", "val", "test"])
    val.set_defaults(func=cmd_validate_detector)

    eval_detector = subparsers.add_parser("evaluate-detector")
    eval_detector.add_argument("--weights", required=True)
    eval_detector.add_argument("--data", default="dataset/yolo_cholecTrack20/cholecTrack20.yaml")
    eval_detector.add_argument("--device", default="0")
    eval_detector.add_argument("--split", default="val", choices=["train", "val", "test"])
    eval_detector.set_defaults(func=cmd_evaluate_detector)

    track = subparsers.add_parser("track")
    track.add_argument("--weights", required=True)
    track.add_argument("--source", required=True)
    track.add_argument("--tracker", default="strongsort")
    track.add_argument(
        "--perspective",
        default="visibility",
        choices=["visibility", "intracorporeal", "intraoperative"],
    )
    track.add_argument("--device", default="0")
    track.add_argument("--conf", type=float, default=0.25)
    track.add_argument("--iou", type=float, default=0.7)
    track.add_argument("--vid-stride", type=int, default=1)
    track.add_argument("--save-video", dest="save", action="store_true")
    track.set_defaults(func=cmd_track)

    benchmark = subparsers.add_parser("benchmark-trackers")
    benchmark.add_argument("--weights", required=True)
    benchmark.add_argument("--source", default="dataset/cholecTrack20/Testing")
    benchmark.add_argument("--gt-source", default="dataset/cholecTrack20")
    benchmark.add_argument("--gt-split", default="Testing")
    benchmark.add_argument("--preset", choices=["smoke"])
    benchmark.add_argument("--videos", nargs="+")
    benchmark.add_argument("--trackers", nargs="+")
    benchmark.add_argument(
        "--perspectives",
        nargs="+",
        choices=["visibility", "intracorporeal", "intraoperative"],
    )
    benchmark.add_argument("--vid-stride", type=int)
    benchmark.add_argument("--jobs", type=int)
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
