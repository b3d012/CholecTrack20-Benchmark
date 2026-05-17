# EfficientDet-D0 -> EA-StrongSORT Experiment Summary

## Research Path

The project first tested YOLO11 as the detector baseline. YOLO11 was documented separately, then the active detector was switched to EfficientDet-D0 because the project goal became testing an EfficientNet-family detector before EA-StrongSORT.

Final contribution path:

```text
EfficientDet-D0 detector -> EA-StrongSORT tracking -> CholecTrack20 multi-perspective evaluation
```

## Selected Detector Run

Selected checkpoint:

```text
results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth
```

Training command:

```powershell
python -m src.train train-detector --config configs/efficientdet_cholecTrack20.yaml --preset tune --epochs 150 --patience 30 --imgsz 640 --batch 6 --accumulation 2 --workers 6 --eval-interval 3 --lr0 0.0002 --device 0 --name efficientdet_d0_640_adamw_lr2e4_long
```

Key hyperparameters:

| Setting | Value |
| --- | --- |
| Model | EfficientDet-D0 (`tf_efficientdet_d0`) |
| Backbone family | EfficientNet-B0 |
| Pretraining | COCO pretrained |
| Classes | 7 surgical tool classes |
| Image size | 640 |
| Optimizer | AdamW |
| Initial learning rate | 0.0002 |
| Final LR factor | 0.01 |
| Weight decay | 0.0001 |
| Batch | 6 |
| Gradient accumulation | 2 |
| Effective batch | 12 |
| Workers | 6 |
| Epochs | 150 |
| Validation interval | every 3 epochs |
| Selected epoch | 63 |

Detector result:

| Detector | AP0.5 | AP0.75 | AP0.5:0.95 | FPS |
| --- | ---: | ---: | ---: | ---: |
| EfficientDet-D0-640-AdamW-lr2e4 | 44.4 | 31.2 | 27.9 | 20.9 |

Full detector tuning history:

```text
EFFICIENTDET_HYPERPARAMETER_COMPARISON.md
EFFICIENTDET_HYPERPARAMETER_COMPARISON.csv
```

## Tracking Run

Tracking command:

```powershell
python -m src.train benchmark-trackers --weights results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth --source dataset/cholecTrack20/Testing --trackers strongsort --perspectives visibility intracorporeal intraoperative --device 0
```

Tracking setup:

| Setting | Value |
| --- | --- |
| Detector | EfficientDet-D0 selected checkpoint |
| Tracker | EA-StrongSORT / StrongSORT |
| ReID weights | `osnet_x0_25_msmt17.pt` |
| Test videos | 8 CholecTrack20 Testing videos |
| Frame stride | 1 |
| Perspectives | visibility, intracorporeal, intraoperative |
| Prediction format | MOT text files |

Tracking prediction output:

| Perspective | Tracker | Sequences | MOT rows | Reported IDs |
| --- | --- | ---: | ---: | ---: |
| visibility | strongsort | 8 | 768555 | 2744 |
| intracorporeal | strongsort | 8 | 768555 | 2744 |
| intraoperative | strongsort | 8 | 768555 | 2744 |

## Tracking Evaluation

Evaluation command:

```powershell
python -m src.train evaluate-tracking --trackers strongsort --perspectives visibility intracorporeal intraoperative --iou-threshold 0.5
```

Evaluation protocol:

| Setting | Value |
| --- | --- |
| Ground truth | CholecTrack20 JSON track IDs |
| Visibility field | `visibility_track` |
| Intracorporeal field | `intracorporeal_track` |
| Intraoperative field | `intraoperative_track` |
| Matching | Class-aware IoU matching |
| IoU threshold | 0.5 |
| Frames evaluated | Annotated CholecTrack20 frames only |

Tracking metrics:

| Perspective | MOTA | IDF1 | MOTP | Precision | Recall | ID switches | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visibility | 20.974 | 24.172 | 82.536 | 68.692 | 58.952 | 3332 | 8059 | 12312 |
| intracorporeal | 19.641 | 12.158 | 82.536 | 68.692 | 58.952 | 3732 | 8059 | 12312 |
| intraoperative | 19.314 | 6.298 | 82.536 | 68.692 | 58.952 | 3830 | 8059 | 12312 |

Class-agnostic diagnostic:

| Perspective | MOTA | IDF1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| visibility | 22.578 | 24.380 | 69.970 | 60.049 |
| intracorporeal | 21.224 | 12.305 | 69.970 | 60.049 |
| intraoperative | 20.844 | 6.373 | 69.970 | 60.049 |

The class-agnostic diagnostic improves only slightly, so the main weakness is not class-label mismatch. The major limitations are missed detections, false positives, and identity fragmentation.

## Interpretation

This does not outperform the strongest published CholecTrack20 tracker baselines, but it provides a documented and reproducible EfficientDet-D0 -> EA-StrongSORT baseline. The work contributes an alternative EfficientNet-family detector path, full detector hyperparameter screening, an EfficientDet-to-StrongSORT adapter, and multi-perspective tracking evaluation.

## Main Output Files

```text
results/logs/efficientdet_paper_detection_metrics.csv
results/logs/tracking_summary.csv
results/logs/tracking_eval_summary.csv
results/logs/tracking_eval_by_sequence.csv
EA_STRONGSORT_TRACKING_RESULTS.md
EA_STRONGSORT_TRACKING_EVALUATION.md
EFFICIENTDET_HYPERPARAMETER_COMPARISON.md
```

