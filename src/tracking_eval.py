"""MOT-style evaluation for CholecTrack20 tracking outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MotBox:
    frame: int
    track_id: int
    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: int


@dataclass
class SequenceStats:
    perspective: str
    tracker: str
    sequence: str
    gt: int = 0
    pred: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    idsw: int = 0
    iou_sum: float = 0.0
    idtp: int = 0
    idfp: int = 0
    idfn: int = 0


def _read_mot(path: Path) -> list[MotBox]:
    rows: list[MotBox] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for parts in reader:
            if len(parts) < 8:
                continue
            try:
                rows.append(
                    MotBox(
                        frame=int(float(parts[0])),
                        track_id=int(float(parts[1])),
                        x=float(parts[2]),
                        y=float(parts[3]),
                        w=float(parts[4]),
                        h=float(parts[5]),
                        conf=float(parts[6]),
                        cls=int(float(parts[7])),
                    )
                )
            except ValueError:
                continue
    return rows


def _by_frame(rows: Iterable[MotBox]) -> dict[int, list[MotBox]]:
    grouped: dict[int, list[MotBox]] = defaultdict(list)
    for row in rows:
        grouped[row.frame].append(row)
    return grouped


def _iou_matrix(gt: list[MotBox], pred: list[MotBox], *, class_aware: bool) -> np.ndarray:
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float32)
    matrix = np.zeros((len(gt), len(pred)), dtype=np.float32)
    for i, gt_box in enumerate(gt):
        gt_x2 = gt_box.x + gt_box.w
        gt_y2 = gt_box.y + gt_box.h
        gt_area = max(gt_box.w, 0.0) * max(gt_box.h, 0.0)
        for j, pred_box in enumerate(pred):
            if class_aware and gt_box.cls != pred_box.cls:
                continue
            pred_x2 = pred_box.x + pred_box.w
            pred_y2 = pred_box.y + pred_box.h
            pred_area = max(pred_box.w, 0.0) * max(pred_box.h, 0.0)
            inter_w = max(0.0, min(gt_x2, pred_x2) - max(gt_box.x, pred_box.x))
            inter_h = max(0.0, min(gt_y2, pred_y2) - max(gt_box.y, pred_box.y))
            intersection = inter_w * inter_h
            union = gt_area + pred_area - intersection
            if union > 0:
                matrix[i, j] = intersection / union
    return matrix


def _evaluate_sequence(
    gt_path: Path,
    pred_path: Path,
    *,
    perspective: str,
    tracker: str,
    iou_threshold: float,
    class_aware: bool,
) -> SequenceStats:
    sequence = gt_path.stem.lower()
    gt_rows = _read_mot(gt_path)
    pred_rows_all = _read_mot(pred_path)
    gt_by_frame = _by_frame(gt_rows)
    annotated_frames = set(gt_by_frame)
    pred_rows = [row for row in pred_rows_all if row.frame in annotated_frames]
    pred_by_frame = _by_frame(pred_rows)

    stats = SequenceStats(
        perspective=perspective,
        tracker=tracker,
        sequence=sequence,
        gt=len(gt_rows),
        pred=len(pred_rows),
    )
    last_pred_for_gt: dict[int, int] = {}
    identity_pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for frame in sorted(annotated_frames):
        gt_frame = gt_by_frame.get(frame, [])
        pred_frame = pred_by_frame.get(frame, [])
        if not gt_frame:
            continue
        if not pred_frame:
            stats.fn += len(gt_frame)
            continue

        ious = _iou_matrix(gt_frame, pred_frame, class_aware=class_aware)
        if ious.size == 0:
            stats.fn += len(gt_frame)
            stats.fp += len(pred_frame)
            continue
        gt_indices, pred_indices = linear_sum_assignment(1.0 - ious)
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for gt_idx, pred_idx in zip(gt_indices, pred_indices):
            iou = float(ious[gt_idx, pred_idx])
            if iou < iou_threshold:
                continue
            gt_box = gt_frame[gt_idx]
            pred_box = pred_frame[pred_idx]
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            stats.tp += 1
            stats.iou_sum += iou
            identity_pair_counts[(gt_box.track_id, pred_box.track_id)] += 1

            previous_pred = last_pred_for_gt.get(gt_box.track_id)
            if previous_pred is not None and previous_pred != pred_box.track_id:
                stats.idsw += 1
            last_pred_for_gt[gt_box.track_id] = pred_box.track_id

        stats.fn += len(gt_frame) - len(matched_gt)
        stats.fp += len(pred_frame) - len(matched_pred)

    stats.idtp = _identity_true_positives(identity_pair_counts)
    stats.idfp = max(0, stats.pred - stats.idtp)
    stats.idfn = max(0, stats.gt - stats.idtp)
    return stats


def _identity_true_positives(pair_counts: dict[tuple[int, int], int]) -> int:
    if not pair_counts:
        return 0
    gt_ids = sorted({pair[0] for pair in pair_counts})
    pred_ids = sorted({pair[1] for pair in pair_counts})
    gt_index = {track_id: idx for idx, track_id in enumerate(gt_ids)}
    pred_index = {track_id: idx for idx, track_id in enumerate(pred_ids)}
    matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for (gt_id, pred_id), count in pair_counts.items():
        matrix[gt_index[gt_id], pred_index[pred_id]] = count
    gt_indices, pred_indices = linear_sum_assignment(-matrix)
    return int(sum(matrix[i, j] for i, j in zip(gt_indices, pred_indices)))


def _aggregate(stats: Iterable[SequenceStats], *, perspective: str, tracker: str) -> dict[str, object]:
    rows = list(stats)
    gt = sum(row.gt for row in rows)
    pred = sum(row.pred for row in rows)
    tp = sum(row.tp for row in rows)
    fp = sum(row.fp for row in rows)
    fn = sum(row.fn for row in rows)
    idsw = sum(row.idsw for row in rows)
    iou_sum = sum(row.iou_sum for row in rows)
    idtp = sum(row.idtp for row in rows)
    idfp = sum(row.idfp for row in rows)
    idfn = sum(row.idfn for row in rows)
    mota = 1.0 - ((fn + fp + idsw) / gt) if gt else 0.0
    motp = iou_sum / tp if tp else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    idf1 = (2 * idtp) / ((2 * idtp) + idfp + idfn) if ((2 * idtp) + idfp + idfn) else 0.0
    return {
        "perspective": perspective,
        "tracker": tracker,
        "sequences": len(rows),
        "gt": gt,
        "pred": pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "idsw": idsw,
        "MOTA": round(mota * 100.0, 3),
        "MOTP": round(motp * 100.0, 3),
        "IDF1": round(idf1 * 100.0, 3),
        "precision": round(precision * 100.0, 3),
        "recall": round(recall * 100.0, 3),
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
    }


def evaluate_tracking(
    gt_root: str | Path,
    pred_root: str | Path,
    perspectives: Iterable[str],
    trackers: Iterable[str],
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    gt_root = Path(gt_root)
    pred_root = Path(pred_root)
    summary_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []

    for perspective in perspectives:
        perspective_dir = gt_root / perspective
        gt_files = sorted(perspective_dir.glob("*.txt"))
        for tracker in trackers:
            tracker_dir = pred_root / tracker
            stats_rows: list[SequenceStats] = []
            for gt_path in gt_files:
                pred_path = tracker_dir / f"{gt_path.stem.lower()}.txt"
                stats = _evaluate_sequence(
                    gt_path,
                    pred_path,
                    perspective=perspective,
                    tracker=tracker,
                    iou_threshold=iou_threshold,
                    class_aware=class_aware,
                )
                stats_rows.append(stats)
                sequence_rows.append(
                    {
                        **stats.__dict__,
                        "MOTA": round(
                            (1.0 - ((stats.fn + stats.fp + stats.idsw) / stats.gt)) * 100.0,
                            3,
                        )
                        if stats.gt
                        else 0.0,
                        "MOTP": round((stats.iou_sum / stats.tp) * 100.0, 3) if stats.tp else 0.0,
                        "IDF1": round(
                            ((2 * stats.idtp) / ((2 * stats.idtp) + stats.idfp + stats.idfn)) * 100.0,
                            3,
                        )
                        if ((2 * stats.idtp) + stats.idfp + stats.idfn)
                        else 0.0,
                    }
                )
            summary_rows.append(_aggregate(stats_rows, perspective=perspective, tracker=tracker))
    return summary_rows, sequence_rows


def write_tracking_evaluation(
    summary_rows: list[dict[str, object]],
    sequence_rows: list[dict[str, object]],
    *,
    summary_path: str | Path,
    sequence_path: str | Path,
) -> None:
    _write_csv(summary_path, summary_rows)
    _write_csv(sequence_path, sequence_rows)


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


