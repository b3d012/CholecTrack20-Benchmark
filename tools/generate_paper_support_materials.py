from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "paper_support"
IMAGE_OUT = OUT / "image_examples"
PRED_DIR = ROOT / "results" / "logs" / "mot_predictions" / "strongsort"
VIDEO_ROOT = ROOT / "dataset" / "cholecTrack20" / "Testing"

TOOL_NAMES = ["Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "Bag"]
COLORS = [
    (0, 180, 255),
    (255, 120, 0),
    (80, 220, 80),
    (255, 80, 80),
    (180, 80, 255),
    (0, 220, 220),
    (255, 180, 80),
]
CHALLENGE_KEYS = {
    "crowded": "Crowded instruments",
    "occluded": "Occlusion",
    "bleeding": "Bleeding",
    "smoke": "Smoke",
    "blurred": "Blur",
    "undercoverage": "Trocar / undercoverage",
    "reflection": "Reflection",
    "stainedlens": "Foul lens",
}


@dataclass
class Candidate:
    score: float
    scenario: str
    video: str
    frame: int
    preds: list[list[float]]
    gt_rows: list[dict]
    tags: list[str]


def read_predictions(path: Path) -> dict[int, list[list[float]]]:
    frames: dict[int, list[list[float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for parts in csv.reader(handle):
            if len(parts) < 8:
                continue
            frame = int(float(parts[0]))
            row = [float(x) for x in parts[1:8]]
            frames.setdefault(frame, []).append(row)
    return frames


def load_annotations(video_stem: str) -> dict[int, list[dict]]:
    json_path = VIDEO_ROOT / video_stem.upper() / f"{video_stem}.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["annotations"].items()}


def tags_for_gt(rows: list[dict]) -> list[str]:
    tags: list[str] = []
    for key, label in CHALLENGE_KEYS.items():
        if any(int(row.get(key, 0)) == 1 for row in rows):
            tags.append(label)
    if not tags:
        tags.append("Clear / standard view")
    return tags


def class_tags(preds: list[list[float]]) -> list[str]:
    classes = sorted({int(row[6]) for row in preds if 0 <= int(row[6]) < len(TOOL_NAMES)})
    return [TOOL_NAMES[i] for i in classes]


def quality_score(preds: list[list[float]], gt_rows: list[dict], tags: list[str]) -> float:
    good = [row for row in preds if row[5] >= 0.45]
    if not good:
        return -1.0
    areas = [max(0.0, row[3] * row[4]) for row in good]
    if max(areas) > 280000 or max(areas) < 3500:
        return -1.0
    avg_conf = sum(row[5] for row in good) / len(good)
    tag_bonus = 0.20 if tags != ["Clear / standard view"] else 0.05
    count_bonus = min(len(good), 4) * 0.06
    area_bonus = min(max(areas) / 120000.0, 1.0) * 0.18
    gt_bonus = min(len(gt_rows), 4) * 0.03
    return avg_conf + tag_bonus + count_bonus + area_bonus + gt_bonus


def build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for pred_path in sorted(PRED_DIR.glob("*.txt")):
        video = pred_path.stem
        predictions = read_predictions(pred_path)
        annotations = load_annotations(video)
        for frame, preds in predictions.items():
            gt_rows = annotations.get(frame, [])
            if not gt_rows:
                continue
            good_preds = [row for row in preds if row[5] >= 0.45]
            if not (1 <= len(good_preds) <= 5):
                continue
            tags = tags_for_gt(gt_rows)
            score = quality_score(good_preds, gt_rows, tags)
            if score < 0:
                continue
            scenario_parts = tags + class_tags(good_preds)
            scenario = " + ".join(scenario_parts[:3])
            candidates.append(Candidate(score, scenario, video, frame, good_preds, gt_rows, tags))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def select_examples(candidates: list[Candidate], n: int = 20) -> list[Candidate]:
    selected: list[Candidate] = []
    used_pairs: set[tuple[str, int]] = set()

    def has_class(name: str):
        class_id = TOOL_NAMES.index(name)
        return lambda item: any(int(row[6]) == class_id for row in item.preds)

    scenario_targets = [
        ("Clear standard tracking case", lambda item: item.tags == ["Clear / standard view"]),
        (
            "Clear multi-tool case with specimen bag",
            lambda item: item.tags == ["Clear / standard view"] and has_class("Bag")(item) and len(item.preds) >= 2,
        ),
        ("Crowded multi-instrument case", lambda item: "Crowded instruments" in item.tags and len(item.preds) >= 4),
        ("Occlusion case", lambda item: "Occlusion" in item.tags),
        ("Smoke case", lambda item: "Smoke" in item.tags),
        ("Bleeding case", lambda item: "Bleeding" in item.tags),
        ("Bleeding and smoke combined case", lambda item: "Bleeding" in item.tags and "Smoke" in item.tags),
        ("Reflection case", lambda item: "Reflection" in item.tags),
        ("Trocar or undercoverage case", lambda item: "Trocar / undercoverage" in item.tags),
        ("Blur case", lambda item: "Blur" in item.tags),
        ("Foul lens case", lambda item: "Foul lens" in item.tags),
        ("Bipolar detection example", has_class("Bipolar")),
        ("Hook detection example", has_class("Hook")),
        ("Scissors detection example", has_class("Scissors")),
        ("Clipper detection example", has_class("Clipper")),
        ("Irrigator detection example", has_class("Irrigator")),
        ("Specimen bag detection example", has_class("Bag")),
        ("High-confidence multi-tool tracking output", lambda item: len(item.preds) >= 4),
        (
            "Mixed difficult-condition tracking output",
            lambda item: len(item.tags) >= 3 and len(item.preds) >= 3,
        ),
        (
            "Clean high-confidence tracking output",
            lambda item: item.tags == ["Clear / standard view"]
            and sum(row[5] for row in item.preds) / len(item.preds) >= 0.95,
        ),
    ]
    for target, predicate in scenario_targets:
        for item in candidates:
            pair = (item.video, item.frame // 500)
            if item in selected or pair in used_pairs:
                continue
            if predicate(item):
                item.scenario = target
                selected.append(item)
                used_pairs.add(pair)
                break
    for item in candidates:
        pair = (item.video, item.frame // 500)
        if item in selected or pair in used_pairs:
            continue
        selected.append(item)
        used_pairs.add(pair)
        if len(selected) >= n:
            break
    return selected[:n]


def get_frame(video: str, frame_no: int):
    video_path = VIDEO_ROOT / video.upper() / f"{video}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_no - 1, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {video_path} frame {frame_no}")
    return frame


def draw_example(item: Candidate, idx: int) -> Path:
    frame = get_frame(item.video, item.frame)
    overlay = frame.copy()
    label_rects: list[tuple[int, int, int, int]] = []
    for pred in sorted(item.preds, key=lambda row: row[5], reverse=True)[:5]:
        track_id, x, y, w, h, conf, cls = pred
        cls_i = int(cls)
        color = COLORS[cls_i % len(COLORS)]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
        tool = TOOL_NAMES[cls_i] if 0 <= cls_i < len(TOOL_NAMES) else f"Class {cls_i}"
        label = f"{tool} ID{int(track_id)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        label_x = min(max(x1, 0), max(0, frame.shape[1] - tw - 12))
        label_y = y1 - 8
        if label_y - th - 8 < 54:
            label_y = min(y2 - 8, y1 + th + 14)
        rect = (label_x, label_y - th - 8, label_x + tw + 8, label_y + 4)
        for _ in range(8):
            overlaps = any(
                not (rect[2] < old[0] or rect[0] > old[2] or rect[3] < old[1] or rect[1] > old[3])
                for old in label_rects
            )
            if not overlaps:
                break
            label_y = min(frame.shape[0] - 58, label_y + th + 14)
            rect = (label_x, label_y - th - 8, label_x + tw + 8, label_y + 4)
        label_rects.append(rect)
        cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), -1)
        cv2.putText(overlay, label, (label_x + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)
    title = f"Example {idx:02d} | {item.video.upper()} frame {item.frame}"
    y_base = frame.shape[0] - 20
    cv2.rectangle(frame, (12, y_base - 36), (12 + 650, y_base + 8), (0, 0, 0), -1)
    cv2.putText(frame, title, (20, y_base - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    out = IMAGE_OUT / f"example_{idx:02d}_{item.video}_frame_{item.frame}.png"
    cv2.imwrite(str(out), frame)
    return out


def write_example_note(item: Candidate, image_path: Path, idx: int) -> Path:
    pred_classes = class_tags(item.preds)
    avg_conf = sum(row[5] for row in item.preds) / len(item.preds)
    note = f"""Example {idx:02d}: {item.scenario}

Image file: {image_path.name}
Video/frame: {item.video.upper()} frame {item.frame}
Predicted detections shown: {len(item.preds)}
Predicted tool classes: {', '.join(pred_classes) if pred_classes else 'unknown'}
Average shown confidence: {avg_conf:.3f}
Ground-truth challenge tags: {', '.join(item.tags)}

How to use this in the paper:
Use this image as a qualitative example for the scenario "{item.scenario}". It is useful for the Results and Discussion section because it shows what the EfficientDet-D0 -> StrongSORT/EA-StrongSORT-oriented pipeline produces on an actual CholecTrack20 test frame.

Interpretation:
The example is relatively clear because the detections have high confidence and visible bounding boxes. It can be used to show that the pipeline works end-to-end and produces readable tool identity outputs. If boxes overlap or labels are crowded, that should be discussed as a realistic surgical-video limitation rather than hidden.

Suggested caption:
Qualitative tracking output on {item.video.upper()} frame {item.frame}. The overlay shows predicted surgical tool class, track ID, and confidence from the EfficientDet-D0 to StrongSORT tracking pipeline under the following condition(s): {', '.join(item.tags)}.
"""
    out = image_path.with_suffix(".txt")
    out.write_text(note, encoding="utf-8")
    return out


def write_idea_file() -> None:
    text = """Paper Additions And Improvement Ideas
====================================

This file lists sections and paragraphs that can be added to the term paper if more depth is needed. The current paper already contains the required structure, but these ideas can make it stronger, especially for a final submission or presentation defense.

1. Expanded Future Work
-----------------------

Add a separate Future Work subsection after the Conclusion or as the final paragraph of the Conclusion.

Suggested content:

Future work should follow a staged plan. First, detector quality should be improved before additional tracker tuning, because tracking-by-detection systems depend heavily on the quality and continuity of frame-level detections. The rare classes, especially irrigator and scissors, need class-aware training strategies such as oversampling, focal loss tuning, copy-paste augmentation, or targeted collection of more examples. Second, the exact EA-StrongSORT ReID branch should be implemented by replacing the OSNet ReID model with an EfficientNetV2 feature extractor using Efficient Channel Attention in deeper layers. Third, GIoU-based association should be compared against the current IoU-based matching strategy. Fourth, tracking should be evaluated per class and per visual challenge so improvements can be linked to specific surgical conditions rather than only global MOTA and IDF1 values.

2. What Could Have Been Done Differently
----------------------------------------

Useful honest discussion:

The project spent significant effort switching detector families after YOLO11 underperformed. A more systematic approach would have been to first reproduce one official CholecTrack20 detector baseline exactly, then introduce EfficientDet-D0 as a controlled comparison. This would make it easier to separate implementation issues from model-family limitations. Another improvement would be to run short 20-30 epoch screening experiments before any 100-150 epoch training run, because the long EfficientDet run took many hours. Finally, more effort could have been spent on data imbalance before tracker benchmarking, since rare-class detection errors directly reduce tracking performance.

3. Stronger Literature Review Angles
------------------------------------

Possible extra paragraphs:

- Explain the difference between detection, segmentation, and tracking in medical video.
- Compare surgical tool tracking with tumor tracking from the EA-StrongSORT paper.
- Discuss why tracking-by-detection is attractive: modular, interpretable, easier to debug, and compatible with standard MOT metrics.
- Discuss why it can fail: detector misses, false positives, box jitter, and identity fragmentation.
- Explain why ReID embeddings are hard in surgery: tools from the same class look similar, metallic reflections change appearance, and only partial tool regions may be visible.

4. Additional Limitations
-------------------------

Potential limitations to add:

- The exact EfficientNetV2/ECA ReID module from EA-StrongSORT was not fully implemented yet.
- The model was trained on limited hardware, which constrained batch size, image size, model scale, and number of experiments.
- The final detector still performs poorly on rare classes, especially irrigator.
- The full benchmark took around 12 hours, limiting the number of tracker configurations that could be tested.
- Qualitative examples show readable tracking outputs, but they also reveal crowded labels and imperfect boxes under surgical visual challenges.

5. Suggested Ablation Study Section
-----------------------------------

If there is time later, add an ablation table like:

Model / Pipeline | Detector AP0.5:0.95 | Visibility MOTA | Visibility IDF1 | Notes
YOLO11 -> StrongSORT | 26.6 | not final / optional | optional | Fast but lower AP
EfficientDet-D0 -> StrongSORT | 27.9 | 20.974 | 24.172 | Current final pipeline
EfficientDet-D0 -> EA-StrongSORT ReID | future | future | future | Planned exact EA extension
EfficientDet-D0 -> EA-StrongSORT + GIoU | future | future | future | Planned association extension

6. Error Analysis Ideas
-----------------------

Add an Error Analysis subsection to Results and Discussion:

- False negatives: missed tools break tracks and reduce recall.
- False positives: incorrect detections create extra tracks and reduce MOTA.
- Box jitter: changing box location from shaft to tip can confuse association.
- Class confusion: similar metallic tools may be assigned wrong labels.
- Rare classes: scarce examples lead to weak AP and unstable tracking.
- Perspective difficulty: visibility is easiest; intraoperative is hardest because identity must persist across longer disappearances.

7. Ethical And Practical Considerations
---------------------------------------

Possible paragraph:

Although the system is intended for research and decision support, it should not be interpreted as a clinical safety device. Surgical AI systems must be validated across institutions, surgeons, camera systems, and patient populations before clinical use. Dataset privacy, licensing, and careful reporting are important because surgical videos are sensitive medical data. In this project, the dataset itself is excluded from GitHub and only summary results, code, and report materials are shared.

8. Better Results Presentation
------------------------------

Add qualitative figures in groups:

- Clear standard cases where tracking works well.
- Crowded/occluded cases where labels are still readable.
- Smoke/reflection/undercoverage cases showing realistic difficulty.
- Failure-prone cases where boxes are large or overlapping.

The generated image bank in reports/paper_support/image_examples contains candidate figures and sidecar notes.
"""
    (OUT / "PAPER_ADDITION_IDEAS.txt").write_text(text, encoding="utf-8")


def main() -> None:
    IMAGE_OUT.mkdir(parents=True, exist_ok=True)
    for old in IMAGE_OUT.glob("example_*"):
        old.unlink()
    for old in [IMAGE_OUT / "index.csv", IMAGE_OUT / "README.md"]:
        if old.exists():
            old.unlink()
    write_idea_file()
    candidates = build_candidates()
    selected = select_examples(candidates, 20)
    index_rows = []
    for idx, item in enumerate(selected, start=1):
        image_path = draw_example(item, idx)
        note_path = write_example_note(item, image_path, idx)
        index_rows.append(
            {
                "example": f"{idx:02d}",
                "image": image_path.name,
                "note": note_path.name,
                "video": item.video.upper(),
                "frame": item.frame,
                "scenario": item.scenario,
                "tags": "; ".join(item.tags),
                "detections": len(item.preds),
                "avg_confidence": f"{sum(row[5] for row in item.preds) / len(item.preds):.3f}",
            }
        )
    with (IMAGE_OUT / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    lines = ["# Paper Image Examples", ""]
    for row in index_rows:
        lines.append(
            f"- Example {row['example']}: `{row['image']}` - {row['scenario']} "
            f"({row['video']} frame {row['frame']}, avg confidence {row['avg_confidence']})."
        )
    (IMAGE_OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} examples to {IMAGE_OUT}")
    print(f"Wrote ideas file to {OUT / 'PAPER_ADDITION_IDEAS.txt'}")


if __name__ == "__main__":
    main()
