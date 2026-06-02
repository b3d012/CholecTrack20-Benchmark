"""Command line entrypoint for training, tracking, ablations, and checks."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from .cholectrack20 import (
    PERSPECTIVE_FIELDS,
    convert_dataset,
    export_mot_ground_truth,
    validate_yolo_labels,
)
from .efficientdet_model import (
    EfficientDetConfig,
    cholec_evaluate_efficientdet,
    evaluate_efficientdet,
    train_efficientdet,
)
from .metrics import detector_metrics_row, write_detection_metrics, write_tracking_metrics
from .strongsort_tracker import TrackingConfig, benchmark_trackers, run_tracker
from .tracking_eval import evaluate_tracking, write_tracking_evaluation
from .utils import append_csv_row, check_environment, load_yaml, save_json, validate_yolo_dataset
from .yolov11_model import DetectorConfig, train_yolov11, validate_yolov11

SMOKE_BENCHMARK = {
    "videos": ["VID01"],
    "perspectives": ["visibility"],
    "trackers": ["botsort", "bytetrack"],
    "vid_stride": 25,
    "jobs": 2,
}

TRAIN_PRESETS = {
    "quick": {
        "epochs": 3,
        "imgsz": 512,
        "workers": 4,
        "patience": 10,
        "fraction": 0.002,
        "plots": False,
        "batch": 2,
        "accumulation": 1,
    },
    "tune": {
        "imgsz": 512,
        "workers": 4,
        "patience": 30,
        "fraction": 1.0,
        "plots": False,
        "batch": 4,
        "accumulation": 2,
    },
    "final": {
        "epochs": 300,
        "imgsz": 512,
        "workers": 4,
        "patience": 50,
        "fraction": 1.0,
        "plots": True,
        "batch": 4,
        "accumulation": 2,
    },
}

DETECTOR_SWEEP = [
    {
        "name": "d0_512_adamw_lr2e4",
        "optimizer": "AdamW",
        "lr0": 0.0002,
        "lrf": 0.01,
        "imgsz": 512,
        "batch": 4,
    },
    {
        "name": "d0_512_adamw_lr1e4",
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.01,
        "imgsz": 512,
        "batch": 4,
    },
    {
        "name": "d0_640_adamw_lr1e4_batch2",
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "lrf": 0.01,
        "imgsz": 640,
        "batch": 2,
    },
    {
        "name": "d0_512_sgd_lr1e3",
        "optimizer": "SGD",
        "lr0": 0.001,
        "lrf": 0.01,
        "imgsz": 512,
        "batch": 4,
    },
]


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


def _append_experiment_log(
    command: str,
    *,
    model: str = "",
    run_name: str = "",
    checkpoint: str = "",
    config_path: str = "",
    split: str = "",
    metrics: dict[str, object] | None = None,
    status: str = "ok",
    notes: str = "",
) -> None:
    metrics = metrics or {}
    append_csv_row(
        "results/logs/experiment_log.csv",
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "command": command,
            "model": model,
            "run_name": run_name,
            "checkpoint": checkpoint,
            "config": config_path,
            "split": split,
            "AP0.5": metrics.get("AP0.5", metrics.get("mAP@0.5", "")),
            "AP0.75": metrics.get("AP0.75", ""),
            "AP0.5:0.95": metrics.get("AP0.5:0.95", metrics.get("mAP@0.5:0.95", "")),
            "precision": metrics.get("precision", ""),
            "recall": metrics.get("recall", ""),
            "fps": metrics.get("fps", ""),
            "status": status,
            "notes": notes,
        },
    )


def _is_yolo_weights(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix == ".pt"


def _efficientdet_config_from_args(args: argparse.Namespace) -> EfficientDetConfig:
    data = load_yaml(args.config) if args.config else {}
    preset = TRAIN_PRESETS.get(args.preset or "", {})
    plots = args.plots if args.plots is not None else preset.get("plots", False)
    return EfficientDetConfig(
        model=args.model or data.get("model", "tf_efficientdet_d0"),
        data=args.data or data.get("data", "dataset/yolo_cholecTrack20/cholecTrack20.yaml"),
        source=args.gt_source if hasattr(args, "gt_source") and args.gt_source else data.get("source", "dataset/cholecTrack20"),
        epochs=args.epochs if args.epochs is not None else int(preset.get("epochs", data.get("epochs", 100))),
        imgsz=args.imgsz if args.imgsz is not None else int(preset.get("imgsz", data.get("imgsz", 512))),
        batch=args.batch if args.batch is not None else int(preset.get("batch", data.get("batch", 4))),
        accumulation=args.accumulation
        if getattr(args, "accumulation", None) is not None
        else int(preset.get("accumulation", data.get("accumulation", 1))),
        device=args.device or str(data.get("device", "0")),
        project=args.project or data.get("project", "results/logs"),
        name=args.name or preset.get("name", data.get("name", "efficientdet_d0_cholectrack20")),
        lr0=args.lr0 if args.lr0 is not None else float(data.get("lr0", 0.0002)),
        lrf=args.lrf if args.lrf is not None else float(data.get("lrf", 0.01)),
        optimizer=args.optimizer or data.get("optimizer", "AdamW"),
        weight_decay=args.weight_decay
        if args.weight_decay is not None
        else float(data.get("weight_decay", 0.0001)),
        warmup_epochs=args.warmup_epochs
        if args.warmup_epochs is not None
        else int(data.get("warmup_epochs", 3)),
        weights_dir=data.get("weights_dir", "results/weights"),
        workers=args.workers if args.workers is not None else int(preset.get("workers", data.get("workers", 4))),
        patience=args.patience if args.patience is not None else int(preset.get("patience", data.get("patience", 30))),
        fraction=args.fraction if args.fraction is not None else float(preset.get("fraction", data.get("fraction", 1.0))),
        plots=bool(plots),
        resume=args.resume,
        amp=not args.no_amp,
        num_classes=int(data.get("num_classes", 7)),
        eval_interval=args.eval_interval
        if getattr(args, "eval_interval", None) is not None
        else int(data.get("eval_interval", 1)),
    )


def _detector_config_from_args(args: argparse.Namespace) -> DetectorConfig:
    data = load_yaml(args.config) if args.config else {}
    preset = TRAIN_PRESETS.get(args.preset or "", {})
    plots = args.plots if args.plots is not None else preset.get("plots", True)
    cache = args.cache if args.cache != "none" else False
    if args.cache is None:
        cache = preset.get("cache", data.get("cache", False))
    return DetectorConfig(
        model=args.model or data.get("model", "yolo11n.pt"),
        data=args.data or data.get("data", "dataset/yolo_cholecTrack20/cholecTrack20.yaml"),
        epochs=args.epochs if args.epochs is not None else int(preset.get("epochs", data.get("epochs", 100))),
        imgsz=args.imgsz if args.imgsz is not None else int(preset.get("imgsz", data.get("imgsz", 640))),
        batch=args.batch if args.batch is not None else int(data.get("batch", 16)),
        device=args.device or str(data.get("device", "0")),
        project=args.project or data.get("project", "results/logs"),
        name=args.name or preset.get("name", data.get("name", "yolov11_cholectrack20")),
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
        cache=cache,
        workers=args.workers if args.workers is not None else int(preset.get("workers", data.get("workers", 8))),
        patience=args.patience if args.patience is not None else int(preset.get("patience", data.get("patience", 100))),
        fraction=args.fraction if args.fraction is not None else float(preset.get("fraction", data.get("fraction", 1.0))),
        plots=bool(plots),
        resume=args.resume,
        amp=not args.no_amp,
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
    if args.detector_backend == "yolo":
        config = _detector_config_from_args(args)
        train_yolov11(config)
        _append_experiment_log(
            "train-detector",
            model=config.model,
            run_name=config.name,
            checkpoint=str(Path(config.weights_dir) / "best.pt"),
            config_path=args.config,
            notes="legacy YOLO detector run",
        )
        return

    config = _efficientdet_config_from_args(args)
    result = train_efficientdet(config)
    _append_experiment_log(
        "train-detector",
        model=config.model,
        run_name=config.name,
        checkpoint=str(Path(config.weights_dir) / "efficientdet_best.pth"),
        config_path=args.config,
        metrics={"AP0.5:0.95": result.get("best_map", "")},
        notes="EfficientDet-D0 detector run",
    )


def _best_weight_path(config: DetectorConfig) -> Path:
    candidates = [
        Path(config.project) / config.name / "weights" / "best.pt",
        Path("runs/detect") / config.project / config.name / "weights" / "best.pt",
        Path(config.weights_dir) / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _write_sweep_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def cmd_sweep_detector(args: argparse.Namespace) -> None:
    rows: list[dict[str, object]] = []
    tag = args.tag or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    out_path = args.out or f"results/logs/efficientdet_{tag}.csv"
    for experiment in DETECTOR_SWEEP:
        run_args = argparse.Namespace(**vars(args))
        run_args.name = f"{tag}_{experiment['name']}"
        run_args.optimizer = experiment["optimizer"]
        run_args.lr0 = experiment["lr0"]
        run_args.lrf = experiment["lrf"]
        run_args.imgsz = experiment["imgsz"]
        run_args.batch = experiment["batch"]
        run_args.epochs = args.epochs
        run_args.plots = args.plots
        run_args.resume = False
        run_args.no_amp = args.no_amp
        config = _efficientdet_config_from_args(run_args)

        print(f"Starting sweep run: {config.name}")
        try:
            result = train_efficientdet(config)
            weights = Path(config.project) / config.name / "weights" / "best.pth"
            metrics = evaluate_efficientdet(
                weights,
                config.data,
                config.source,
                args.split,
                config.device,
                imgsz=args.eval_imgsz,
                batch=args.eval_batch,
                half=args.half,
            )
            row = {
                "name": config.name,
                "status": "ok",
                "weights": str(weights),
                "optimizer": config.optimizer,
                "lr0": config.lr0,
                "lrf": config.lrf,
                "imgsz": config.imgsz,
                "batch": config.batch,
                "epochs": config.epochs,
                "precision": "",
                "recall": "",
                "mAP@0.5": metrics.get("AP0.5"),
                "mAP@0.5:0.95": metrics.get("AP0.5:0.95"),
                "error": "",
            }
            _append_experiment_log(
                "sweep-detector",
                model=config.model,
                run_name=config.name,
                checkpoint=str(weights),
                config_path=args.config,
                split=args.split,
                metrics=metrics,
                notes=f"sweep candidate {experiment['name']}",
            )
        except Exception as exc:
            row = {
                "name": config.name,
                "status": "failed",
                "weights": "",
                "optimizer": config.optimizer,
                "lr0": config.lr0,
                "lrf": config.lrf,
                "imgsz": config.imgsz,
                "batch": config.batch,
                "epochs": config.epochs,
                "precision": "",
                "recall": "",
                "mAP@0.5": "",
                "mAP@0.5:0.95": "",
                "error": str(exc),
            }
            _append_experiment_log(
                "sweep-detector",
                model=config.model,
                run_name=config.name,
                config_path=args.config,
                split=args.split,
                status="failed",
                notes=str(exc),
            )
        rows.append(row)
        _write_sweep_rows(out_path, rows)
        print(f"{row['name']}: {row['status']} mAP@0.5:0.95={row['mAP@0.5:0.95']}")
    print(f"Sweep summary written to {out_path}")


def cmd_validate_detector(args: argparse.Namespace) -> None:
    if args.detector_backend == "yolo" or _is_yolo_weights(args.weights):
        result = validate_yolov11(
            args.weights,
            args.data,
            args.device,
            args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            half=args.half,
            plots=args.plots,
        )
        row = detector_metrics_row(result, args.split)
        log_metrics = row
        model_name = "YOLO"
    else:
        metrics = evaluate_efficientdet(
            args.weights,
            args.data,
            args.gt_source,
            args.split,
            args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            half=args.half,
        )
        row = {
            "split": args.split,
            "precision": "",
            "recall": "",
            "mAP@0.5": metrics.get("AP0.5"),
            "mAP@0.5:0.95": metrics.get("AP0.5:0.95"),
        }
        log_metrics = metrics
        model_name = "EfficientDet-D0"
    write_detection_metrics("results/logs/detection_metrics.csv", row)
    _append_experiment_log(
        "evaluate-detector",
        model=model_name,
        checkpoint=args.weights,
        split=args.split,
        metrics=log_metrics,
    )
    for key, value in row.items():
        print(f"{key}: {value}")


def cmd_evaluate_detector(args: argparse.Namespace) -> None:
    cmd_validate_detector(args)


def cmd_cholec_detection_eval(args: argparse.Namespace) -> None:
    if args.detector_backend == "yolo" or _is_yolo_weights(args.weights):
        raise SystemExit(
            "This CholecTrack20 detection-eval CLI is reserved for EfficientDet checkpoints. "
            "Use evaluate-detector for YOLO checkpoints."
        )
    row = cholec_evaluate_efficientdet(
        args.weights,
        args.data,
        args.gt_source,
        args.split,
        args.device,
        args.imgsz,
        args.batch,
        args.half,
        args.model_name,
        args.output,
    )
    _append_experiment_log(
        "cholec-detection-eval",
        model=args.model_name,
        checkpoint=args.weights,
        split=args.split,
        metrics={
            "AP0.5": row.get("AP0.5", ""),
            "AP0.75": row.get("AP0.75", ""),
            "AP0.5:0.95": row.get("AP0.5:0.95", ""),
            "fps": row.get("FPS", ""),
        },
    )
    for key, value in row.items():
        print(f"{key}: {value}")


def cmd_track(args: argparse.Namespace) -> None:
    result = run_tracker(
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
    _append_experiment_log(
        "track",
        model="EfficientDet-D0" if not _is_yolo_weights(args.weights) else "YOLO",
        run_name=f"tracking_{args.tracker}",
        checkpoint=args.weights,
        metrics=result if isinstance(result, dict) else {},
        notes=f"tracker={args.tracker}; perspective={args.perspective}",
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
        _append_experiment_log(
            "benchmark-trackers",
            model="EfficientDet-D0" if not _is_yolo_weights(args.weights) else "YOLO",
            run_name=f"{row['perspective']}_{row['tracker']}",
            checkpoint=args.weights,
            metrics=row,
            status=str(row.get("status", "ok")),
            notes=f"videos={args.videos or 'all'}; vid_stride={args.vid_stride}",
        )
        print(
            f"{row['perspective']} {row['tracker']}: {row['status']} "
            f"({row['detections']} detections, {row['ids']} ids)"
        )


def cmd_evaluate_tracking(args: argparse.Namespace) -> None:
    perspectives = args.perspectives or ["visibility", "intracorporeal", "intraoperative"]
    trackers = args.trackers or ["strongsort"]
    invalid = [perspective for perspective in perspectives if perspective not in PERSPECTIVE_FIELDS]
    if invalid:
        raise SystemExit(f"Unsupported perspective(s): {', '.join(invalid)}")

    export_mot_ground_truth(
        args.gt_source,
        args.gt_dir,
        perspectives,
        split_name=args.gt_split,
    )
    summary_rows, sequence_rows = evaluate_tracking(
        args.gt_dir,
        args.pred_dir,
        perspectives,
        trackers,
        iou_threshold=args.iou_threshold,
        class_aware=not args.class_agnostic,
    )
    write_tracking_evaluation(
        summary_rows,
        sequence_rows,
        summary_path=args.summary_out,
        sequence_path=args.sequence_out,
    )
    for row in summary_rows:
        _append_experiment_log(
            "evaluate-tracking",
            model="tracking-eval",
            run_name=f"{row['perspective']}_{row['tracker']}",
            metrics={"precision": row.get("precision", ""), "recall": row.get("recall", "")},
            notes=(
                f"MOTA={row.get('MOTA', '')}; IDF1={row.get('IDF1', '')}; "
                f"IoU={args.iou_threshold}; class_aware={not args.class_agnostic}"
            ),
        )
        print(
            f"{row['perspective']} {row['tracker']}: "
            f"MOTA={row['MOTA']} IDF1={row['IDF1']} "
            f"precision={row['precision']} recall={row['recall']}"
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
    train.add_argument("--config", default="configs/efficientdet_cholecTrack20.yaml")
    train.add_argument("--detector-backend", choices=["efficientdet", "yolo"], default="efficientdet")
    train.add_argument("--preset", choices=["quick", "tune", "final"])
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
    train.add_argument("--accumulation", type=int)
    train.add_argument("--eval-interval", type=int)
    train.add_argument("--cache", choices=["none", "ram", "disk"])
    train.add_argument("--workers", type=int)
    train.add_argument("--patience", type=int)
    train.add_argument("--fraction", type=float)
    train.add_argument("--plots", dest="plots", action="store_true")
    train.add_argument("--no-plots", dest="plots", action="store_false")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--no-amp", action="store_true")
    train.set_defaults(plots=None)
    train.set_defaults(func=cmd_train_detector)

    sweep = subparsers.add_parser("sweep-detector")
    sweep.add_argument("--config", default="configs/efficientdet_cholecTrack20.yaml")
    sweep.add_argument("--detector-backend", choices=["efficientdet"], default="efficientdet")
    sweep.add_argument("--preset", choices=["quick", "tune", "final"], default="tune")
    sweep.add_argument("--data")
    sweep.add_argument("--model")
    sweep.add_argument("--epochs", type=int, default=50)
    sweep.add_argument("--device")
    sweep.add_argument("--project")
    sweep.add_argument("--weight-decay", type=float)
    sweep.add_argument("--warmup-epochs", type=int)
    sweep.add_argument("--accumulation", type=int)
    sweep.add_argument("--eval-interval", type=int)
    sweep.add_argument("--cache", choices=["none", "ram", "disk"])
    sweep.add_argument("--workers", type=int)
    sweep.add_argument("--patience", type=int)
    sweep.add_argument("--fraction", type=float)
    sweep.add_argument("--plots", dest="plots", action="store_true")
    sweep.add_argument("--no-plots", dest="plots", action="store_false")
    sweep.add_argument("--no-amp", action="store_true")
    sweep.add_argument("--split", default="val", choices=["train", "val", "test"])
    sweep.add_argument("--eval-imgsz", type=int, default=640)
    sweep.add_argument("--eval-batch", type=int, default=16)
    sweep.add_argument("--gt-source", default="dataset/cholecTrack20")
    sweep.add_argument("--half", action="store_true")
    sweep.add_argument("--tag")
    sweep.add_argument("--out")
    sweep.set_defaults(plots=None, resume=False, func=cmd_sweep_detector)

    val = subparsers.add_parser("validate-detector")
    val.add_argument("--weights", required=True)
    val.add_argument("--data", default="dataset/yolo_cholecTrack20/cholecTrack20.yaml")
    val.add_argument("--gt-source", default="dataset/cholecTrack20")
    val.add_argument("--detector-backend", choices=["efficientdet", "yolo", "auto"], default="auto")
    val.add_argument("--device", default="0")
    val.add_argument("--split", default="val", choices=["train", "val", "test"])
    val.add_argument("--imgsz", type=int, default=640)
    val.add_argument("--batch", type=int, default=16)
    val.add_argument("--half", action="store_true")
    val.add_argument("--plots", action="store_true")
    val.set_defaults(func=cmd_validate_detector)

    eval_detector = subparsers.add_parser("evaluate-detector")
    eval_detector.add_argument("--weights", required=True)
    eval_detector.add_argument("--data", default="dataset/yolo_cholecTrack20/cholecTrack20.yaml")
    eval_detector.add_argument("--gt-source", default="dataset/cholecTrack20")
    eval_detector.add_argument("--detector-backend", choices=["efficientdet", "yolo", "auto"], default="auto")
    eval_detector.add_argument("--device", default="0")
    eval_detector.add_argument("--split", default="val", choices=["train", "val", "test"])
    eval_detector.add_argument("--imgsz", type=int, default=640)
    eval_detector.add_argument("--batch", type=int, default=16)
    eval_detector.add_argument("--half", action="store_true")
    eval_detector.add_argument("--plots", action="store_true")
    eval_detector.set_defaults(func=cmd_evaluate_detector)

    cholec_eval = subparsers.add_parser("cholec-detection-eval")
    cholec_eval.add_argument("--weights", required=True)
    cholec_eval.add_argument("--data", default="dataset/yolo_cholecTrack20/cholecTrack20.yaml")
    cholec_eval.add_argument("--gt-source", default="dataset/cholecTrack20")
    cholec_eval.add_argument("--detector-backend", choices=["efficientdet", "yolo", "auto"], default="auto")
    cholec_eval.add_argument("--device", default="0")
    cholec_eval.add_argument("--split", default="val", choices=["train", "val", "test"])
    cholec_eval.add_argument("--imgsz", type=int)
    cholec_eval.add_argument("--batch", type=int, default=4)
    cholec_eval.add_argument("--half", action="store_true")
    cholec_eval.add_argument("--model-name", default="EfficientDet-D0")
    cholec_eval.add_argument("--output", default="results/logs/efficientdet_cholec_detection_metrics")
    cholec_eval.set_defaults(func=cmd_cholec_detection_eval)

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

    tracking_eval = subparsers.add_parser("evaluate-tracking")
    tracking_eval.add_argument("--gt-source", default="dataset/cholecTrack20")
    tracking_eval.add_argument("--gt-split", default="Testing")
    tracking_eval.add_argument("--gt-dir", default="results/logs/mot_ground_truth")
    tracking_eval.add_argument("--pred-dir", default="results/logs/mot_predictions")
    tracking_eval.add_argument("--trackers", nargs="+", default=["strongsort"])
    tracking_eval.add_argument(
        "--perspectives",
        nargs="+",
        choices=["visibility", "intracorporeal", "intraoperative"],
    )
    tracking_eval.add_argument("--iou-threshold", type=float, default=0.5)
    tracking_eval.add_argument("--class-agnostic", action="store_true")
    tracking_eval.add_argument("--summary-out", default="results/logs/tracking_eval_summary.csv")
    tracking_eval.add_argument("--sequence-out", default="results/logs/tracking_eval_by_sequence.csv")
    tracking_eval.set_defaults(func=cmd_evaluate_tracking)

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
