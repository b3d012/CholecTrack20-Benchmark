"""Paper-style detection evaluation for CholecTrack20.

This script computes the detector table fields used in the CholecTrack20 paper:
overall AP at 0.5, 0.75, 0.5:0.95, per-tool AP@0.5, challenge AP@0.5,
and prediction FPS.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


CLASS_NAMES = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "Bag",
]

CHALLENGES = {
    "Bleeding": "bleeding",
    "Blur": "blurred",
    "Smoke": "smoke",
    "Crowded": "crowded",
    "Occluded": "occluded",
    "Reflection": "reflection",
    "Foul Lens": "stainedlens",
    "Trocar": "undercoverage",
}

SPLIT_DIRS = {
    "train": "Training",
    "val": "Validation",
    "test": "Testing",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_abs_tlwh(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.0:
        x *= width
        w *= width
        y *= height
        h *= height
    return [x, y, w, h]


def build_coco_gt(source: Path, yolo_root: Path, split: str) -> dict[str, Any]:
    split_dir = source / SPLIT_DIRS[split]
    image_dir = yolo_root / "images" / split
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id_by_stem: dict[str, int] = {}
    ann_id = 1

    for video_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        json_path = next(iter(sorted(video_dir.glob("*.json"))), None)
        if json_path is None:
            continue
        dataset = _load_json(json_path)
        video = dataset["video"]
        video_name = video["name"]
        width = int(video["width"])
        height = int(video["height"])

        for frame_key in sorted(dataset["annotations"], key=lambda item: int(item)):
            frame_id = int(frame_key)
            stem = f"{video_name}_{frame_id:06d}"
            image_path = image_dir / f"{stem}.png"
            if not image_path.exists():
                continue
            if stem not in image_id_by_stem:
                image_id = len(images) + 1
                image_id_by_stem[stem] = image_id
                images.append(
                    {
                        "id": image_id,
                        "file_name": image_path.as_posix(),
                        "width": width,
                        "height": height,
                        "stem": stem,
                    }
                )
            image_id = image_id_by_stem[stem]
            for source_ann in dataset["annotations"][frame_key]:
                bbox = _to_abs_tlwh(source_ann["tool_bbox"], width, height)
                area = float(bbox[2] * bbox[3])
                ann = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(source_ann["instrument"]) + 1,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": int(source_ann.get("iscrowd", 0)),
                }
                for field in CHALLENGES.values():
                    ann[field] = int(source_ann.get(field, 0))
                annotations.append(ann)
                ann_id += 1

    return {
        "info": {"description": "CholecTrack20 YOLO evaluation subset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": idx + 1, "name": name.lower().replace("bag", "specimen-bag")}
            for idx, name in enumerate(CLASS_NAMES)
        ],
    }


def predict_yolo(
    weights: Path,
    images: list[dict[str, Any]],
    imgsz: int,
    batch: int,
    device: str,
    half: bool,
) -> tuple[list[dict[str, Any]], float]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    image_id_by_path = {image["file_name"]: int(image["id"]) for image in images}
    paths = list(image_id_by_path)
    predictions: list[dict[str, Any]] = []

    started = time.perf_counter()
    chunk_size = max(1, batch)
    for start in range(0, len(paths), chunk_size):
        chunk = paths[start : start + chunk_size]
        results = model.predict(
            source=chunk,
            imgsz=imgsz,
            batch=len(chunk),
            device=device,
            half=half,
            conf=0.001,
            iou=0.7,
            max_det=300,
            verbose=False,
            stream=False,
        )
        for result, source_path in zip(results, chunk):
            image_id = image_id_by_path[source_path]
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for box, conf, cls_id in zip(xyxy, confs, classes):
                x1, y1, x2, y2 = [float(value) for value in box]
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(cls_id) + 1,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(conf),
                    }
                )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return predictions, len(paths) / elapsed


def _subset_gt(gt: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    return {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "images": [image for image in gt["images"] if int(image["id"]) in image_ids],
        "annotations": [ann for ann in gt["annotations"] if int(ann["image_id"]) in image_ids],
        "categories": gt["categories"],
    }


def _challenge_image_ids(gt: dict[str, Any], field: str) -> set[int]:
    return {int(ann["image_id"]) for ann in gt["annotations"] if int(ann.get(field, 0)) == 1}


def coco_ap(
    gt: dict[str, Any],
    predictions: list[dict[str, Any]],
    *,
    cat_id: int | None = None,
) -> dict[str, float]:
    from contextlib import redirect_stdout
    from io import StringIO

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not gt["images"] or not gt["annotations"]:
        return {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}

    image_ids = {int(image["id"]) for image in gt["images"]}
    predictions = [prediction for prediction in predictions if int(prediction["image_id"]) in image_ids]

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions or [])
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = [int(image["id"]) for image in gt["images"]]
    if cat_id is not None:
        evaluator.params.catIds = [cat_id]
    with redirect_stdout(StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
    }


def _pct(value: float) -> str:
    if value != value:
        return ""
    return f"{value * 100:.1f}"


def write_outputs(row: dict[str, str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(row)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)

    md_path = output.with_suffix(".md")
    md_path.write_text(
        "| " + " | ".join(columns) + " |\n"
        + "| " + " | ".join("---" for _ in columns) + " |\n"
        + "| " + " | ".join(row[column] for column in columns) + " |\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paper-style CholecTrack20 detector metrics.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", default="dataset/cholecTrack20")
    parser.add_argument("--yolo-root", default="dataset/yolo_cholecTrack20")
    parser.add_argument("--split", default="val", choices=sorted(SPLIT_DIRS))
    parser.add_argument("--model-name", default="YOLO11")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output", default="results/logs/yolo11_paper_detection_metrics")
    args = parser.parse_args()

    gt = build_coco_gt(Path(args.source), Path(args.yolo_root), args.split)
    predictions, fps = predict_yolo(
        Path(args.weights),
        gt["images"],
        args.imgsz,
        args.batch,
        args.device,
        args.half,
    )

    overall = coco_ap(gt, predictions)
    row: dict[str, str] = {
        "Detection model": args.model_name,
        "AP0.5": _pct(overall["AP50"]),
        "AP0.75": _pct(overall["AP75"]),
        "AP0.5:0.95": _pct(overall["AP"]),
    }

    for idx, class_name in enumerate(CLASS_NAMES, start=1):
        row[class_name] = _pct(coco_ap(gt, predictions, cat_id=idx)["AP50"])

    for challenge_name, field in CHALLENGES.items():
        image_ids = _challenge_image_ids(gt, field)
        challenge_gt = _subset_gt(gt, image_ids)
        row[challenge_name] = _pct(coco_ap(challenge_gt, predictions)["AP50"])

    row["FPS"] = f"{fps:.1f}"
    write_outputs(row, Path(args.output))
    for key, value in row.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
