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

For YOLO training, create or export a dataset YAML file with this shape:

```yaml
path: dataset/cholecTrack20
train: images/train
val: images/val
test: images/test
names:
  0: grasper
  1: bipolar
  2: hook
  3: scissors
  4: clipper
  5: irrigator
```

Then validate the layout:

```bash
python -m src.train validate-data --data dataset/cholecTrack20.yaml
```

## Training

```bash
python -m src.train train-detector \
  --data dataset/cholecTrack20.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

Weights and logs are written to `results/weights/` and `results/logs/`.

## Tracking

Run tracking with the trained detector:

```bash
python -m src.train track \
  --weights results/weights/best.pt \
  --source dataset/cholecTrack20/videos/val \
  --tracker strongsort \
  --device 0
```

Benchmark supported tracker modes:

```bash
python -m src.train benchmark-trackers \
  --weights results/weights/best.pt \
  --source dataset/cholecTrack20/videos/val \
  --trackers botsort bytetrack ocsort strongsort \
  --device 0
```

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
