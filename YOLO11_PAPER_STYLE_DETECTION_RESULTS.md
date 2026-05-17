# YOLO11 Paper-Style Detection Results

This file records the YOLO11 detector in the same table format as the CholecTrack20 detector comparison table.

## Evaluation Setup

- Detector: `YOLO11-img768-SGD`
- Checkpoint: `runs/detect/results/logs/final_yolo11_img768_sgd/weights/best.pt`
- Split: `val`
- Image size: `768`
- Batch: `4`
- Half precision: enabled
- Evaluator: `src/paper_detection_eval.py`
- Output files:
  - `results/logs/yolo11_img768_sgd_paper_detection_metrics.csv`
  - `results/logs/yolo11_img768_sgd_paper_detection_metrics.md`

Command:

```powershell
python -m src.paper_detection_eval --weights runs/detect/results/logs/final_yolo11_img768_sgd/weights/best.pt --split val --imgsz 768 --batch 4 --device 0 --half --model-name YOLO11-img768-SGD --output results/logs/yolo11_img768_sgd_paper_detection_metrics
```

## Result Row

| Detection model | AP0.5 | AP0.75 | AP0.5:0.95 | Grasper | Bipolar | Hook | Scissors | Clipper | Irrigator | Bag | Bleeding | Blur | Smoke | Crowded | Occluded | Reflection | Foul Lens | Trocar | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11-img768-SGD | 40.3 | 29.6 | 26.6 | 42.1 | 45.8 | 50.0 | 23.6 | 59.3 | 3.4 | 57.8 | 15.1 | 100.0 | 43.5 | 22.8 | 46.5 |  | 12.7 | 22.5 | 62.9 |

## Notes

- Values are percentages, matching the style of the reference detector table.
- `Reflection` is blank because the validation split did not provide a usable reflection subset for this evaluator.
- Challenge AP is computed on validation images that contain at least one annotation with the corresponding CholecTrack20 challenge flag.
- `Trocar` is mapped to the CholecTrack20 JSON field `undercoverage`.
- These numbers are for our validation split, so they are useful for internal comparison and the term paper. They should not be presented as the official CholecTrack20 hidden test benchmark unless the same official evaluation split/protocol is used.

