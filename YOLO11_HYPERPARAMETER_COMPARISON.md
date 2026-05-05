# YOLO11 Hyperparameter Comparison

This file summarizes the YOLO11 detector experiments run before moving on to EA-StrongSORT tracking.

Metrics are taken from each run's `results.csv`, using the best epoch by `metrics/mAP50-95(B)`.

## Serious Runs

| Run | Key settings | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `final_yolo11_img768_sgd` | SGD, `imgsz=768`, `batch=8`, `lr0=0.01`, `patience=30` | 42 | 0.5595 | 0.4683 | 0.4077 | **0.2670** | Best strict-box score. Slightly better localization, but not a large gain. |
| `final_yolo11_patience30` | SGD, `imgsz=640`, `batch=16`, `lr0=0.01`, `patience=30` | 31 | **0.5700** | 0.4715 | **0.4108** | 0.2631 | Best overall balance and best mAP50. Strong baseline. |
| `final_yolo11_adamw_lr001` | AdamW, `imgsz=640`, `batch=16`, `lr0=0.001`, `patience=40` | 44 | 0.5573 | **0.4716** | 0.4035 | 0.2519 | Did not beat SGD. Some class-level changes, but worse overall. |
| `sweep_20260503_130508_sweep_sgd_lr001_img640` | SGD, `imgsz=640`, `batch=16`, `lr0=0.01`, sweep run | 31 | 0.5638 | 0.4521 | 0.4048 | 0.2627 | Very close to the main SGD baseline, but slightly worse. |

## Quick And Diagnostic Runs

| Run | Key settings | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `quick_yolo_test` | SGD, `imgsz=384`, `batch=16`, `epochs=10`, `fraction=0.35` | 10 | 0.3910 | 0.3338 | 0.2867 | 0.1608 | Useful for pipeline testing only. Not a final-quality detector. |
| `batch_test_24` | Batch-size test | 1 | 0.3048 | 0.3122 | 0.2261 | 0.1326 | Diagnostic run only; not enough training to compare quality. |

## Current Best Choice

Use `final_yolo11_img768_sgd` if the priority is the best `mAP50-95`:

```powershell
runs/detect/results/logs/final_yolo11_img768_sgd/weights/best.pt
```

Use `final_yolo11_patience30` if the priority is best overall balance and `mAP50`:

```powershell
runs/detect/results/logs/final_yolo11_patience30/weights/best.pt
```

For the next tracking stage, the recommended detector is `final_yolo11_img768_sgd` because it has the best strict localization score, even though the improvement is small.

## Main Takeaways

- SGD performed better than AdamW for this dataset/config.
- Increasing image size from 640 to 768 gave a small `mAP50-95` improvement: `0.2631 -> 0.2670`.
- The gain from `imgsz=768` is real but small, so more YOLO training may have diminishing returns.
- The weak class remains `irrigator`; this likely needs data/class-balance work more than optimizer changes.
- Move to EA-StrongSORT benchmarking now, then come back to YOLO only if tracking results are limited by detector quality.

