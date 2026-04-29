"""Tracking wrappers for StrongSORT-style and benchmark MOT runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .yolov11_model import load_yolov11


ULTRALYTICS_TRACKER_CONFIGS = {
    "botsort": "botsort.yaml",
    "bytetrack": "bytetrack.yaml",
}


@dataclass(frozen=True)
class TrackingConfig:
    weights: str
    source: str
    tracker: str = "strongsort"
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
    device: str = "0",
) -> dict[str, object]:
    results: dict[str, object] = {}
    for tracker in trackers:
        run_config = TrackingConfig(
            weights=weights,
            source=source,
            tracker=tracker,
            device=device,
            name=f"tracking_{tracker}",
        )
        results[tracker] = run_tracker(run_config)
    return results

