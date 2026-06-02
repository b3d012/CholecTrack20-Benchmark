# EA-StrongSORT Tracking Results

Detector used:

```text
EfficientDet-D0 640, AdamW, lr0=0.0002
results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth
```

Benchmark command:

```powershell
python -m src.train benchmark-trackers --weights results/logs/efficientdet_d0_640_adamw_lr2e4_long/weights/best.pth --source dataset/cholecTrack20/Testing --trackers strongsort --perspectives visibility intracorporeal intraoperative --device 0
```

## Summary

| Perspective | Tracker | Status | Sequences | MOT rows | Reported IDs | Reported speed |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| visibility | strongsort | ok | 8 | 768555 | 2744 | 17.518 |
| intracorporeal | strongsort | ok | 8 | 768555 | 2744 | 17.518 |
| intraoperative | strongsort | ok | 8 | 768555 | 2744 | 17.518 |

The three perspectives have identical prediction counts because the detector/tracker predictions are computed once and reused against each perspective. The perspective-specific difference should appear only after evaluating predictions against the corresponding ground-truth track IDs.

## Per-Video Prediction Counts

| Video | MOT rows | Unique predicted IDs | First predicted frame | Last predicted frame |
| --- | ---: | ---: | ---: | ---: |
| vid01 | 63916 | 868 | 135 | 45175 |
| vid06 | 129533 | 2068 | 3 | 68141 |
| vid07 | 153942 | 2744 | 3 | 118224 |
| vid111 | 74698 | 1177 | 1094 | 53650 |
| vid12 | 47305 | 803 | 237 | 29393 |
| vid25 | 93157 | 1518 | 3 | 58871 |
| vid39 | 133510 | 1251 | 117 | 78748 |
| vid92 | 72494 | 1397 | 3 | 53088 |

## Output Files

```text
results/logs/tracking_visibility.csv
results/logs/tracking_intracorporeal.csv
results/logs/tracking_intraoperative.csv
results/logs/tracking_summary.csv
results/logs/mot_predictions/strongsort/
results/logs/tracking_eval_summary.csv
results/logs/tracking_eval_by_sequence.csv
EA_STRONGSORT_TRACKING_EVALUATION.md
```
