# Tracking Evaluation Results

Metrics are computed on annotated CholecTrack20 frames only, using IoU >= 0.5 and class-agnostic matching. This is a diagnostic run only; the class-aware results are the main tracking metrics.

Command:

```powershell
python -m src.train evaluate-tracking --trackers strongsort --perspectives visibility intracorporeal intraoperative --iou-threshold 0.5 --class-agnostic --summary-out results/logs/tracking_eval_summary_class_agnostic.csv --sequence-out results/logs/tracking_eval_by_sequence_class_agnostic.csv --markdown-out EA_STRONGSORT_TRACKING_EVALUATION_CLASS_AGNOSTIC.md
```

Detector checkpoint:

```text
results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth
```

| perspective | tracker | sequences | MOTA | IDF1 | MOTP | precision | recall | idsw | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visibility | strongsort | 8 | 22.578 | 24.38 | 82.455 | 69.97 | 60.049 | 3509 | 7730 | 11983 |
| intracorporeal | strongsort | 8 | 21.224 | 12.305 | 82.455 | 69.97 | 60.049 | 3915 | 7730 | 11983 |
| intraoperative | strongsort | 8 | 20.844 | 6.373 | 82.455 | 69.97 | 60.049 | 4029 | 7730 | 11983 |
