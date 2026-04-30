"""Tracking wrappers for StrongSORT-style and benchmark MOT runs."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .yolov11_model import load_yolov11


ULTRALYTICS_TRACKER_CONFIGS = {
    "botsort": "botsort.yaml",
    "bytetrack": "bytetrack.yaml",
}

TRACKER_ALIASES = {
    "strongsort": "strongsort",
    "ea-strongsort": "strongsort",
    "ea_strongsort": "strongsort",
    "ocsort": "ocsort",
    "botsort": "botsort",
    "bytetrack": "bytetrack",
}


@dataclass(frozen=True)
class TrackingConfig:
    weights: str
    source: str
    tracker: str = "strongsort"
    perspective: str = "visibility"
    device: str = "0"
    conf: float = 0.25
    iou: float = 0.7
    project: str = "results/logs"
    name: str = "tracking"
    save: bool = True


class EAStrongSORTTracker:
    """Thin runner for detector-driven multi-object tracking experiments."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.detector = load_yolov11(config.weights)

    def run(self) -> object:
        normalized = TRACKER_ALIASES.get(self.config.tracker.lower(), self.config.tracker.lower())
        if normalized in {"strongsort", "ocsort"}:
            return self._run_boxmot_or_raise(normalized)
        tracker = self._resolve_tracker(self.config.tracker)
        return self.detector.track(
            source=self.config.source,
            tracker=tracker,
            device=self.config.device,
            conf=self.config.conf,
            iou=self.config.iou,
            project=self.config.project,
            name=self.config.name,
            save=self.config.save,
            exist_ok=True,
        )

    def run_to_mot(self, output_dir: str | Path) -> dict[str, object]:
        started = time.perf_counter()
        normalized = TRACKER_ALIASES.get(self.config.tracker.lower(), self.config.tracker.lower())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if normalized in {"strongsort", "ocsort"}:
            self._run_boxmot_or_raise(normalized)
            return {
                "status": "ran",
                "sequences": 0,
                "detections": 0,
                "ids": 0,
                "fps": None,
                "error": "BoxMOT output parsing is backend-dependent; inspect BoxMOT run folder.",
            }

        sources = self._iter_sources(Path(self.config.source))
        sequences = 0
        detections = 0
        ids: set[int] = set()
        for source in sources:
            sequence_name = source.stem if source.is_file() else source.name
            rows = self._track_source(source)
            sequences += 1
            detections += len(rows)
            for row in rows:
                ids.add(int(row[1]))
            self._write_mot(output_dir / f"{sequence_name}.txt", rows)
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "status": "ok",
            "sequences": sequences,
            "detections": detections,
            "ids": len(ids),
            "fps": round(detections / elapsed, 3),
            "error": "",
        }

    def _track_source(self, source: Path) -> list[list[float]]:
        tracker = self._resolve_tracker(self.config.tracker)
        results = self.detector.track(
            source=str(source),
            tracker=tracker,
            device=self.config.device,
            conf=self.config.conf,
            iou=self.config.iou,
            save=self.config.save,
            stream=True,
            persist=True,
            project=self.config.project,
            name=self.config.name,
            exist_ok=True,
        )
        rows: list[list[float]] = []
        for frame_idx, result in enumerate(results, start=1):
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None:
                continue
            xywh = boxes.xywh.cpu().numpy()
            track_ids = boxes.id.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()
            for box, track_id, conf, cls_id in zip(xywh, track_ids, confs, classes):
                x_center, y_center, width, height = [float(value) for value in box]
                rows.append(
                    [
                        frame_idx,
                        int(track_id),
                        x_center - width / 2.0,
                        y_center - height / 2.0,
                        width,
                        height,
                        float(conf),
                        int(cls_id),
                        -1,
                    ]
                )
        return rows

    @staticmethod
    def _write_mot(path: Path, rows: list[list[float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for frame, track_id, x, y, w, h, conf, cls_id, marker in rows:
                writer.writerow(
                    [
                        int(frame),
                        int(track_id),
                        f"{x:.2f}",
                        f"{y:.2f}",
                        f"{w:.2f}",
                        f"{h:.2f}",
                        f"{conf:.4f}",
                        int(cls_id),
                        int(marker),
                    ]
                )

    @staticmethod
    def _iter_sources(source: Path) -> list[Path]:
        if source.is_file():
            return [source]
        videos = sorted(
            item for item in source.rglob("*") if item.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if videos:
            return videos
        images = sorted(
            item for item in source.rglob("*") if item.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        return images if images else [source]

    def _run_boxmot_or_raise(self, tracker_name: str) -> object:
        try:
            import boxmot  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"{tracker_name} requires BoxMOT. Install the environment with "
                "`conda env create -f environment.yml`, then rerun tracking."
            ) from exc
        executable = shutil.which("boxmot")
        if executable is None:
            executable = sys.executable
            command = [executable, "-m", "boxmot"]
        else:
            command = [executable]
        command.extend(
            [
                "track",
                "--detector",
                self.config.weights,
                "--tracker",
                tracker_name,
                "--source",
                self.config.source,
                "--device",
                self.config.device,
                "--save",
            ]
        )
        if tracker_name == "strongsort":
            command.extend(["--reid", "osnet_x0_25_msmt17.pt"])
        subprocess.run(command, check=True)
        return {"status": "ran"}

    @staticmethod
    def _resolve_tracker(tracker: str) -> str:
        normalized = tracker.lower()
        if normalized in ULTRALYTICS_TRACKER_CONFIGS:
            return ULTRALYTICS_TRACKER_CONFIGS[normalized]
        if normalized in {"strongsort", "ocsort"}:
            config_path = Path("configs") / "trackers" / f"{normalized}.yaml"
            return str(config_path) if config_path.exists() else normalized
        raise ValueError(f"Unsupported tracker: {tracker}")


def run_tracker(config: TrackingConfig) -> object:
    return EAStrongSORTTracker(config).run()


def benchmark_trackers(
    weights: str,
    source: str,
    trackers: Iterable[str],
    perspectives: Iterable[str] = ("visibility",),
    device: str = "0",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for perspective in perspectives:
        for tracker in trackers:
            run_config = TrackingConfig(
                weights=weights,
                source=source,
                tracker=tracker,
                perspective=perspective,
                device=device,
                name=f"tracking_{perspective}_{tracker}",
            )
            try:
                output_dir = Path("results/logs") / "mot_predictions" / perspective / tracker
                summary = EAStrongSORTTracker(run_config).run_to_mot(output_dir)
            except Exception as exc:
                summary = {
                    "status": "failed",
                    "sequences": 0,
                    "detections": 0,
                    "ids": 0,
                    "fps": None,
                    "error": str(exc),
                }
            rows.append({"perspective": perspective, "tracker": tracker, **summary})
    return rows
