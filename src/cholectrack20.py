"""CholecTrack20 conversion and ground-truth export utilities."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


SPLIT_MAP = {
    "Training": "train",
    "Validation": "val",
    "Testing": "test",
}

PERSPECTIVE_FIELDS = {
    "visibility": "visibility_track",
    "intracorporeal": "intracorporeal_track",
    "intraoperative": "intraoperative_track",
}


@dataclass(frozen=True)
class ConversionSummary:
    split: str
    videos: int
    images: int
    labels: int
    objects: int


def _video_json(video_dir: Path) -> Path:
    matches = sorted(video_dir.glob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No annotation JSON found in {video_dir}")
    return matches[0]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _tool_names(dataset: dict) -> list[str]:
    tools = dataset["categories"]["tools"]
    return [tool["name"] for tool in sorted(tools, key=lambda item: item["id"])]


def _normalize_tlwh(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in bbox]
    if max(abs(x), abs(y), abs(w), abs(h)) > 1.0:
        x /= width
        w /= width
        y /= height
        h /= height
    xc = x + (w / 2.0)
    yc = y + (h / 2.0)
    return (
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(w, 0.0), 1.0),
        min(max(h, 0.0), 1.0),
    )


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _extract_frame(video_path: Path, frame_id: int, dst: Path) -> bool:
    if dst.exists():
        return True
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to extract CholecTrack20 test frames.") from exc

    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    for candidate in (frame_id, max(frame_id - 1, 0)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate)
        ok, frame = cap.read()
        if ok and frame is not None:
            cap.release()
            return bool(cv2.imwrite(str(dst), frame))
    cap.release()
    return False


def _source_image(video_dir: Path, frame_id: int) -> Path | None:
    frame_path = video_dir / "Frames" / f"{frame_id:06d}.png"
    return frame_path if frame_path.exists() else None


def convert_dataset(source: str | Path, out: str | Path) -> list[ConversionSummary]:
    source = Path(source)
    out = Path(out)
    summaries: list[ConversionSummary] = []
    names: list[str] | None = None

    for original_split, split in SPLIT_MAP.items():
        split_dir = source / original_split
        video_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        image_count = 0
        label_count = 0
        object_count = 0

        for video_dir in video_dirs:
            dataset = _load_json(_video_json(video_dir))
            names = names or _tool_names(dataset)
            video_name = dataset["video"]["name"]
            width = int(dataset["video"]["width"])
            height = int(dataset["video"]["height"])
            annotations = dataset["annotations"]
            mp4 = next(iter(sorted(video_dir.glob("*.mp4"))), None)

            for frame_key in sorted(annotations, key=lambda key: int(key)):
                frame_id = int(frame_key)
                stem = f"{video_name}_{frame_id:06d}"
                image_dst = out / "images" / split / f"{stem}.png"
                label_dst = out / "labels" / split / f"{stem}.txt"

                image_src = _source_image(video_dir, frame_id)
                if image_src is not None:
                    _safe_link_or_copy(image_src, image_dst)
                elif mp4 is not None:
                    if not _extract_frame(mp4, frame_id, image_dst):
                        raise RuntimeError(f"Could not extract frame {frame_id} from {mp4}")
                else:
                    raise FileNotFoundError(f"No image or video found for {video_dir} frame {frame_id}")

                rows: list[str] = []
                for ann in annotations[frame_key]:
                    cls_id = int(ann["instrument"])
                    x, y, w, h = _normalize_tlwh(ann["tool_bbox"], width, height)
                    rows.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                label_dst.parent.mkdir(parents=True, exist_ok=True)
                label_dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                image_count += 1
                label_count += 1
                object_count += len(rows)

        summaries.append(
            ConversionSummary(
                split=split,
                videos=len(video_dirs),
                images=image_count,
                labels=label_count,
                objects=object_count,
            )
        )

    if not names:
        raise RuntimeError(f"No CholecTrack20 videos found under {source}")
    dataset_yaml = {
        "path": str(out.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(names)},
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cholecTrack20.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)
    return summaries


def validate_yolo_labels(root: str | Path) -> list[str]:
    root = Path(root)
    problems: list[str] = []
    for label_file in root.glob("labels/*/*.txt"):
        for line_no, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{label_file}:{line_no} expected 5 columns")
                continue
            cls_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
            if cls_id < 0 or cls_id > 6:
                problems.append(f"{label_file}:{line_no} invalid class {cls_id}")
            if any(value < 0.0 or value > 1.0 for value in values):
                problems.append(f"{label_file}:{line_no} bbox values outside 0-1")
    return problems


def export_mot_ground_truth(
    source: str | Path,
    out: str | Path,
    perspectives: Iterable[str],
    split_name: str = "Testing",
) -> dict[str, int]:
    source = Path(source)
    out = Path(out)
    counts: dict[str, int] = {}
    video_dirs = sorted(path for path in (source / split_name).iterdir() if path.is_dir())
    for perspective in perspectives:
        field = PERSPECTIVE_FIELDS[perspective]
        total = 0
        for video_dir in video_dirs:
            dataset = _load_json(_video_json(video_dir))
            video_name = dataset["video"]["name"]
            width = int(dataset["video"]["width"])
            height = int(dataset["video"]["height"])
            rows: list[str] = []
            for frame_key in sorted(dataset["annotations"], key=lambda key: int(key)):
                frame_id = int(frame_key)
                for ann in dataset["annotations"][frame_key]:
                    track_id = int(ann.get(field, -1))
                    if track_id < 0:
                        continue
                    x, y, w, h = ann["tool_bbox"]
                    if max(abs(float(x)), abs(float(y)), abs(float(w)), abs(float(h))) <= 1.0:
                        x, w = float(x) * width, float(w) * width
                        y, h = float(y) * height, float(h) * height
                    cls_id = int(ann["instrument"])
                    rows.append(
                        f"{frame_id},{track_id},{float(x):.2f},{float(y):.2f},"
                        f"{float(w):.2f},{float(h):.2f},1,{cls_id},1"
                    )
            dst = out / perspective / f"{video_name}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            total += len(rows)
        counts[perspective] = total
    return counts
