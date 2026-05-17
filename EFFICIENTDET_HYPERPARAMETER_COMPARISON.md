# EfficientDet-D0 Hyperparameter Comparison

This file records the EfficientDet-D0 detector experiments used before moving to EA-StrongSORT tracking.

The active detector pipeline is now:

```text
EfficientDet-D0 -> EA-StrongSORT
```

YOLO11 results are preserved separately in:

```text
YOLO11_HYPERPARAMETER_COMPARISON.md
YOLO11_PAPER_STYLE_DETECTION_RESULTS.md
```

## Runs

| Selected | Run | Total epochs | Best epoch | Key settings | AP0.5 | AP0.75 | AP0.5:0.95 | FPS | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| yes | `efficientdet_d0_640_adamw_lr2e4_long` | 150 | 63 | D0, AdamW, `lr0=0.0002`, `imgsz=640`, `batch=6`, `accumulation=2`, `workers=6`, `eval_interval=3` | 44.4 | **31.3** | **27.9** | 20.9 | Best EfficientDet run so far. Use `weights/best.pth`; later epochs declined slightly. |
| no | `efficientdet_d0_512_adamw_lr2e4_batch12_workers6_eval3` | 100 | 96 | D0, AdamW, `lr0=0.0002`, `lrf=0.01`, `imgsz=512`, `batch=12`, `workers=6`, `eval_interval=3` | 46.6 | 28.5 | 27.2 | 28.8 | First serious EfficientDet run. Slightly beats YOLO11 on AP0.5:0.95. |
| no | `efficientdet_d0_quick_smoke` | 1 | 1 | D0, `imgsz=512`, tiny quick preset | 0.0 | 0.0 | 0.0 | 30.7 | Pipeline sanity check only. |
| no | `screen_effdet_d0_lr1e4` | 30 | 25 | D0, AdamW, `lr0=0.0001`, `lrf=0.01`, `imgsz=512`, `batch=12`, `workers=6`, `eval_interval=5` | 41.7 | 26.3 | 24.8 |  | Lower-LR screen. Not better than the `lr0=0.0002` run, so do not spend a full 100 epochs on this exact setup. |
| no | `screen_effdet_d0_640_lr1e4` | 30 | 30 | D0, AdamW, `lr0=0.0001`, `imgsz=640` | 41.2 | 28.9 | 25.7 |  | Higher AP0.75/AP0.5:0.95 than the 512 lower-LR screen, but slower per epoch. |
| no | `screen_effdet_d0_640_lr2e4` | 30 | 30 | D0, AdamW, `lr0=0.0002`, `imgsz=640` | 42.9 | 29.3 | 26.6 |  | Strongest 30-epoch screen. Nearly matched the previous 100-epoch best by epoch 30. |
| no | `screen_effdet_d0_sgd_lr1e3` | 30 | 20 | D0, SGD, `lr0=0.001` | 37.3 | 22.9 | 21.8 |  | SGD screen underperformed clearly. Reject for now. |
| no | `final_efficientdet_d0` | 300 |  | D0, final preset, `epochs=300`, `patience=30` |  |  |  |  | Final detector candidate placeholder. |

## Selected Detector

The selected detector for tracking is:

```text
results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth
```

Training command:

```powershell
python -m src.train train-detector --config configs/efficientdet_cholecTrack20.yaml --preset tune --epochs 150 --patience 30 --imgsz 640 --batch 6 --accumulation 2 --workers 6 --eval-interval 3 --lr0 0.0002 --device 0 --name efficientdet_d0_640_adamw_lr2e4_long
```

Paper-style detector evaluation command:

```powershell
python -m src.train paper-detection-eval --weights results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth --split val --half --model-name EfficientDet-D0-640-AdamW-lr2e4
```

Paper-style detector row:

| Detection model | AP0.5 | AP0.75 | AP0.5:0.95 | Grasper | Bipolar | Hook | Scissors | Clipper | Irrigator | Bag | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EfficientDet-D0-640-AdamW-lr2e4 | 44.4 | 31.2 | 27.9 | 47.3 | 50.8 | 60.3 | 33.4 | 60.5 | 0.6 | 57.6 | 20.9 |

## Notes

- Each run should append to `results/logs/experiment_log.csv`.
- Paper-style detector rows should be generated with `paper-detection-eval`.
- Current best checkpoint shortcut: `results/weights/efficientdet_best.pth`.
- Current best per-run checkpoint: `results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth`.
- Paper-style metrics for the selected run are saved in `results/logs/efficientdet_paper_detection_metrics.csv` and `.md`.
- The selected checkpoint was then used for EA-StrongSORT tracking and tracking evaluation.
