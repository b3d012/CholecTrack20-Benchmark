# Tracking Evaluation Results

Metrics are computed on annotated CholecTrack20 frames only, using IoU >= 0.5 and class-aware matching.

Command:

```powershell
python -m src.train evaluate-tracking --trackers strongsort --perspectives visibility intracorporeal intraoperative --iou-threshold 0.5
```

Detector checkpoint:

```text
results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth
```

| perspective | tracker | sequences | MOTA | IDF1 | MOTP | precision | recall | idsw | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visibility | strongsort | 8 | 20.974 | 24.172 | 82.536 | 68.692 | 58.952 | 3332 | 8059 | 12312 |
| intracorporeal | strongsort | 8 | 19.641 | 12.158 | 82.536 | 68.692 | 58.952 | 3732 | 8059 | 12312 |
| intraoperative | strongsort | 8 | 19.314 | 6.298 | 82.536 | 68.692 | 58.952 | 3830 | 8059 | 12312 |
