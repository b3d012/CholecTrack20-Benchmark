# EA-StrongSORT on CholecTrack20

This project implements a training and evaluation scaffold for surgical tool detection and multi-object tracking on CholecTrack20. The detector baseline uses Ultralytics YOLOv11, while tracking is prepared around StrongSORT-style tracking and benchmark comparisons with Bot-SORT, ByteTrack, and OCSORT.

## Project Phases

1. Phase 1 - Dataset understanding: inspect CholecTrack20 labels, classes, and video/frame layout.
2. Phase 2 - Environment setup: create the Conda environment, verify CUDA, and run a smoke test.
3. Phase 3 - Training and testing: train YOLOv11 detection and run multi-object tracking.
4. Phase 4 - Results and analysis: compare trackers and run ablations for GIoU, EfficientNetV2, and ECA.
5. Phase 5 - Presentation: prepare reports, slides, and a demo.

## Repository Layout

```text
DeepLearningProject/
├── dataset/
│   ├── cholecTrack20/
│   └── annotations/
├── src/
│   ├── yolov11_model.py
│   ├── strongsort_tracker.py
│   ├── train.py
│   └── utils.py
├── results/
│   ├── logs/
│   └── weights/
├── reports/
└── slides/
```

## Setup

```bash
conda env create -f environment.yml
conda activate deep_learning_project
python -m src.train check-env
```

The environment uses PyTorch 2.x because current Ultralytics YOLOv11 packages expect a modern PyTorch stack. If your GPU driver does not support CUDA 12.1, edit `environment.yml` to match your installed CUDA runtime.

## Dataset Preparation

Place CholecTrack20 files under:

```text
dataset/cholecTrack20/
dataset/annotations/
```

Convert the official CholecTrack20 JSON annotations into YOLO format:

```bash
python -m src.train prepare-data \
  --source dataset/cholecTrack20 \
  --out dataset/yolo_cholecTrack20
```

This creates `dataset/yolo_cholecTrack20/cholecTrack20.yaml`, YOLO labels, and extracted test frames where needed. The seven classes are `grasper`, `bipolar`, `hook`, `scissors`, `clipper`, `irrigator`, and `specimen-bag`.

Then validate the layout:

```bash
python -m src.train validate-data --data dataset/yolo_cholecTrack20/cholecTrack20.yaml
```

## Training

Default hyperparameters are defined in `configs/yolov11_cholecTrack20.yaml`:

- `epochs`: 100, tunable up to 300
- `batch`: 16
- `imgsz`: 640
- `lr0`: 0.01
- `lrf`: 0.01
- `optimizer`: SGD
- `weight_decay`: 0.0005
- `warmup_epochs`: 3

```bash
python -m src.train train-detector --config configs/yolov11_cholecTrack20.yaml
```

Weights and logs are written to `results/weights/` and `results/logs/`.

Evaluate detector metrics:

```bash
python -m src.train evaluate-detector \
  --weights results/weights/best.pt \
  --split val
```

The exported detector metrics are `Precision`, `Recall`, `mAP@0.5`, and `mAP@0.5:0.95` in `results/logs/detection_metrics.csv`.

## Tracking

Run tracking with the trained detector:

```bash
python -m src.train track \
  --weights results/weights/best.pt \
  --source dataset/cholecTrack20/Testing \
  --tracker strongsort \
  --perspective visibility \
  --device 0
```

Benchmark supported tracker modes across all CholecTrack20 tracking perspectives:

```bash
python -m src.train benchmark-trackers \
  --weights results/weights/best.pt \
  --source dataset/cholecTrack20/Testing \
  --trackers botsort bytetrack ocsort strongsort \
  --perspectives visibility intracorporeal intraoperative \
  --device 0
```

This writes:

- `results/logs/tracking_visibility.csv`
- `results/logs/tracking_intracorporeal.csv`
- `results/logs/tracking_intraoperative.csv`
- `results/logs/tracking_summary.csv`

## Ablation Study

```bash
python -m src.train ablate \
  --data dataset/cholecTrack20.yaml \
  --device 0
```

The ablation scaffold records runs for:

- Baseline YOLOv11
- GIoU-enabled loss configuration
- EfficientNetV2 feature extractor option
- Efficient Channel Attention (ECA)
- EfficientNetV2 + ECA

## Baseline References

- StrongSORT upstream implementation: https://github.com/dyhBUPT/StrongSORT
- Ultralytics YOLO documentation: https://docs.ultralytics.com

## GitHub Notes

Large assets are excluded by `.gitignore`: datasets, trained weights, model exports, logs, and environment-local files. Commit code, configuration, documentation, reports, and slides; keep datasets and generated outputs outside version control or use Git LFS only when explicitly needed.
