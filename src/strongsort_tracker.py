"""Tracking wrappers for StrongSORT-style and benchmark MOT runs."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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

BOXMOT_DETECTOR_TOKENS = {
    "sam",
    "rtdetr_v2_r101vd",
    "rtdetr_v2_r18vd",
    "rtdetr_v2_r50vd",
    "yolo11",
    "yolo12",
    "yolo26",
    "yolo8",
    "yolo9",
    "yolo10",
    "yolov8",
    "yolov9",
    "yolov10",
    "yolox_l",
    "yolox_m",
    "yolox_n",
    "yolox_s",
    "yolox_x",
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
    save: bool = False
    vid_stride: int = 1


class EAStrongSORTTracker:
    """Thin runner for detector-driven multi-object tracking experiments."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.detector = None

    def _detector(self):
        if self.detector is None:
            self.detector = load_yolov11(self.config.weights)
        return self.detector

    def run(self) -> object:
        normalized = TRACKER_ALIASES.get(self.config.tracker.lower(), self.config.tracker.lower())
        if normalized in {"strongsort", "ocsort"}:
            return self._run_boxmot_or_raise(normalized)
        tracker = self._resolve_tracker(self.config.tracker)
        return self._detector().track(
            source=self.config.source,
            tracker=tracker,
            device=self.config.device,
            conf=self.config.conf,
            iou=self.config.iou,
            project=self.config.project,
            name=self.config.name,
            save=self.config.save,
            vid_stride=self.config.vid_stride,
            exist_ok=True,
        )

    def run_to_mot(self, output_dir: str | Path) -> dict[str, object]:
        started = time.perf_counter()
        normalized = TRACKER_ALIASES.get(self.config.tracker.lower(), self.config.tracker.lower())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sources = self._iter_sources(Path(self.config.source))
        sequences = 0
        detections = 0
        ids: set[int] = set()
        for source in sources:
            sequence_name = source.stem if source.is_file() else source.name
            if normalized in {"strongsort", "ocsort"}:
                path = self._run_boxmot_source_to_mot(normalized, source, output_dir / f"{sequence_name}.txt")
                rows = self._read_mot_summary(path)
            else:
                rows = self._track_source(source)
                self._write_mot(output_dir / f"{sequence_name}.txt", rows)
            sequences += 1
            detections += len(rows)
            for row in rows:
                ids.add(int(row[1]))
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
        results = self._detector().track(
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
            vid_stride=self.config.vid_stride,
        )
        rows: list[list[float]] = []
        for result_idx, result in enumerate(results, start=1):
            frame_idx = ((result_idx - 1) * max(self.config.vid_stride, 1)) + 1
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

    @staticmethod
    def _filter_sources(sources: Sequence[Path], videos: Iterable[str] | None) -> list[Path]:
        if not videos:
            return list(sources)
        requested = {video.lower() for video in videos}
        filtered = [
            source
            for source in sources
            if source.stem.lower() in requested or source.parent.name.lower() in requested
        ]
        missing = sorted(requested - {source.stem.lower() for source in filtered} - {source.parent.name.lower() for source in filtered})
        if missing:
            raise FileNotFoundError(f"Requested video(s) not found: {', '.join(missing)}")
        return filtered

    def _run_boxmot_or_raise(self, tracker_name: str) -> object:
        try:
            import boxmot  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"{tracker_name} requires BoxMOT. Install the environment with "
                "`conda env create -f environment.yml`, then rerun tracking."
            ) from exc
        command = self._boxmot_command()

        env = self._boxmot_env()
        detector = self._boxmot_detector_path(Path(self.config.weights))
        sources = self._iter_sources(Path(self.config.source))
        for source in sources:
            run_name = self.config.name
            if len(sources) > 1:
                run_name = f"{self.config.name}_{source.stem}"

            if self.config.save:
                run_command = self._boxmot_cli_command(command, detector, tracker_name, source, run_name)
            else:
                output = Path(self.config.project) / "track" / run_name / "tracks.txt"
                run_command = self._boxmot_stream_command(detector, tracker_name, source, output)
            if tracker_name == "strongsort":
                run_command.extend(["--reid", "osnet_x0_25_msmt17.pt"])
            subprocess.run(run_command, check=True, env=env)
        return {"status": "ran", "sources": len(sources)}

    def _run_boxmot_source_to_mot(self, tracker_name: str, source: Path, output: Path) -> Path:
        try:
            import boxmot  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"{tracker_name} requires BoxMOT. Install the environment with "
                "`conda env create -f environment.yml`, then rerun tracking."
            ) from exc

        output.parent.mkdir(parents=True, exist_ok=True)
        env = self._boxmot_env()
        detector = self._boxmot_detector_path(Path(self.config.weights))
        run_command = self._boxmot_stream_command(detector, tracker_name, source, output)
        if tracker_name == "strongsort":
            run_command.extend(["--reid", "osnet_x0_25_msmt17.pt"])
        subprocess.run(run_command, check=True, env=env)
        return output

    def _boxmot_cli_command(self, command: list[str], detector: str, tracker_name: str, source: Path, run_name: str) -> list[str]:
        return [
            *command,
            "track",
            "--detector",
            detector,
            "--tracker",
            tracker_name,
            "--source",
            str(source),
            "--device",
            self.config.device,
            "--conf",
            str(self.config.conf),
            "--iou",
            str(self.config.iou),
            "--project",
            self.config.project,
            "--name",
            run_name,
            "--exist-ok",
            "--save",
            "--save-txt",
            "--vid-stride",
            str(self.config.vid_stride),
        ]

    def _boxmot_stream_command(self, detector: str, tracker_name: str, source: Path, output: str | Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "src.boxmot_stream",
            "--detector",
            detector,
            "--tracker",
            tracker_name,
            "--source",
            str(source),
            "--output",
            str(output),
            "--device",
            self.config.device,
            "--conf",
            str(self.config.conf),
            "--iou",
            str(self.config.iou),
            "--vid-stride",
            str(self.config.vid_stride),
        ]

    @staticmethod
    def _read_mot_summary(path: Path) -> list[list[float]]:
        if not path.exists():
            return []
        rows: list[list[float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 2:
                    continue
                rows.append([float(parts[0]), float(parts[1])])
        return rows

    @staticmethod
    def _boxmot_detector_path(weights: Path) -> str:
        weights_name = weights.name.lower()
        if any(token in weights_name for token in BOXMOT_DETECTOR_TOKENS):
            return str(weights)

        alias = weights.with_name(f"yolo11_{weights.name}")
        if not alias.exists() or alias.stat().st_size != weights.stat().st_size:
            try:
                if alias.exists():
                    alias.unlink()
                os.link(weights, alias)
            except OSError:
                shutil.copy2(weights, alias)
        return str(alias)

    @staticmethod
    def _boxmot_command() -> list[str]:
        executable = shutil.which("boxmot")
        if executable:
            return [executable]

        candidates = [
            Path(sys.executable).with_name("boxmot.exe"),
            Path(sys.executable).parent / "Scripts" / "boxmot.exe",
            Path(sys.executable).with_name("boxmot"),
            Path(sys.executable).parent / "bin" / "boxmot",
        ]
        for candidate in candidates:
            if candidate.exists():
                return [str(candidate)]

        return [sys.executable, "-c", "from boxmot.engine.cli import main; main()"]

    @staticmethod
    def _boxmot_env() -> dict[str, str]:
        env = os.environ.copy()
        home = (Path("results/logs/boxmot_home")).resolve()
        (home / ".cache" / "gdown").mkdir(parents=True, exist_ok=True)
        (home / "Ultralytics").mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)
        env["YOLO_CONFIG_DIR"] = str(home / "Ultralytics")
        return env

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
    videos: Iterable[str] | None = None,
    vid_stride: int = 1,
    jobs: int = 1,
) -> list[dict[str, object]]:
    perspectives = list(perspectives)
    trackers = list(trackers)
    base_sources = EAStrongSORTTracker._filter_sources(EAStrongSORTTracker._iter_sources(Path(source)), videos)
    rows_by_tracker: dict[str, dict[str, object]] = {}

    def run_one_tracker(tracker: str) -> tuple[str, dict[str, object]]:
        started = time.perf_counter()
        tracker_dir = Path("results/logs") / "mot_predictions" / tracker
        sequences = 0
        detections = 0
        ids: set[int] = set()
        try:
            for video_source in base_sources:
                run_config = TrackingConfig(
                    weights=weights,
                    source=str(video_source),
                    tracker=tracker,
                    device=device,
                    name=f"benchmark_{tracker}_{video_source.stem}",
                    vid_stride=max(int(vid_stride), 1),
                )
                summary = EAStrongSORTTracker(run_config).run_to_mot(tracker_dir)
                sequences += int(summary.get("sequences", 0) or 0)
                detections += int(summary.get("detections", 0) or 0)
                # IDs are only unique within a video; this mirrors the old aggregate behavior closely enough.
                ids.update(range(int(summary.get("ids", 0) or 0)))
            elapsed = max(time.perf_counter() - started, 1e-9)
            return tracker, {
                "status": "ok",
                "sequences": sequences,
                "detections": detections,
                "ids": len(ids),
                "fps": round(detections / elapsed, 3),
                "error": "",
            }
        except Exception as exc:
            return tracker, {
                "status": "failed",
                "sequences": sequences,
                "detections": detections,
                "ids": len(ids),
                "fps": None,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max(int(jobs), 1)) as executor:
        futures = [executor.submit(run_one_tracker, tracker) for tracker in trackers]
        for future in as_completed(futures):
            tracker, summary = future.result()
            rows_by_tracker[tracker] = summary

    rows: list[dict[str, object]] = []
    for perspective in perspectives:
        for tracker in trackers:
            rows.append({"perspective": perspective, "tracker": tracker, **rows_by_tracker[tracker]})
    return rows
