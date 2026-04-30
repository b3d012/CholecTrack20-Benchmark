"""Metric export helpers for detector and tracking experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


DETECTION_COLUMNS = ["split", "precision", "recall", "mAP@0.5", "mAP@0.5:0.95"]
TRACKING_COLUMNS = [
    "perspective",
    "tracker",
    "status",
    "sequences",
    "detections",
    "ids",
    "fps",
    "error",
]


def _write_rows(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def detector_metrics_row(result: Any, split: str) -> dict[str, Any]:
    box = getattr(result, "box", None)
    return {
        "split": split,
        "precision": getattr(box, "mp", None),
        "recall": getattr(box, "mr", None),
        "mAP@0.5": getattr(box, "map50", None),
        "mAP@0.5:0.95": getattr(box, "map", None),
    }


def write_detection_metrics(path: str | Path, row: dict[str, Any]) -> None:
    _write_rows(path, [row], DETECTION_COLUMNS)


def write_tracking_metrics(path: str | Path, rows: list[dict[str, Any]]) -> None:
    _write_rows(path, rows, TRACKING_COLUMNS)
