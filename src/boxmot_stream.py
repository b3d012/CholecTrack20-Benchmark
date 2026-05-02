"""Memory-safe BoxMOT tracking runner.

BoxMOT's CLI text export materializes an entire video before writing tracks.
For long CholecTrack20 videos, stream results to disk frame by frame instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream BoxMOT tracks to MOT text.")
    parser.add_argument("--detector", required=True)
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--reid")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from boxmot.data import iter_source
    from boxmot.engine.results import Results
    from boxmot.engine.tracker import _build_detector, _build_reid, _build_tracker
    from boxmot.utils.mot_utils import write_mot_results

    class StreamingResults(Results):
        def __init__(self, *init_args, vid_stride: int = 1, **kwargs) -> None:
            super().__init__(*init_args, **kwargs)
            self.vid_stride = max(int(vid_stride), 1)

        def _iter_frames(self):
            yield from iter_source(self.source, vid_stride=self.vid_stride)

        def original_frame_id(self, frame_idx: int) -> int:
            return ((int(frame_idx) - 1) * self.vid_stride) + 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    detector = _build_detector(args, detector_spec=args.detector, classes=None)
    tracker = _build_tracker(args, tracker_spec=args.tracker)
    reid = _build_reid(args, tracker, reid_spec=args.reid, tracker_spec=args.tracker)

    run = StreamingResults(args.source, detector, reid, tracker, verbose=args.verbose, vid_stride=args.vid_stride)
    run._cache_results = False
    for track_result in run:
        mot_rows = track_result.to_mot()
        if mot_rows.size:
            mot_rows = np.array(mot_rows, copy=True)
            mot_rows[:, 0] = run.original_frame_id(track_result.frame_idx)
        write_mot_results(output, mot_rows)


if __name__ == "__main__":
    main()
