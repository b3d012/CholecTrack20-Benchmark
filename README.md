# EA-StrongSORT on CholecTrack20

This project implements a training and evaluation scaffold for surgical tool detection and multi-object tracking on CholecTrack20. YOLO11 was tested first and kept as a documented baseline, but the active detector pipeline now uses EfficientDet-D0, an EfficientNet-B0 + BiFPN detector, before EA-StrongSORT tracking.

## Project Phases

1. Phase 1 - Dataset understanding: inspect CholecTrack20 labels, classes, and video/frame layout.
2. Phase 2 - Environment setup: create the Conda environment, verify CUDA, and run a smoke test.
3. Phase 3 - Training and testing: train EfficientDet-D0 detection and run multi-object tracking.
4. Phase 4 - Results and analysis: compare EfficientDet-D0 against the YOLO11 baseline and evaluate EA-StrongSORT.
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

This workstation stores the project Conda environment and package cache on `D:` to avoid filling the Windows system drive:

```text
D:\conda_envs\deep_learning_project
D:\conda_pkgs
D:\temp\deep_learning_project
```

Conda is configured so `conda activate deep_learning_project` resolves to the `D:` environment. The environment also sets `TEMP`, `TMP`, and `ULTRALYTICS_CONFIG_DIR` to `D:\temp\deep_learning_project` when activated.

Recreate the environment from scratch if an old `deep_learning_project` environment already exists:

```powershell
conda deactivate
conda env remove -n deep_learning_project -y
conda env create -f environment.yml
conda activate deep_learning_project
where python
```

Run these smoke checks before training or evaluation:

```powershell
python -c "import _ctypes, torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip check
python -m src.train check-env
python -c "import effdet, timm; print('efficientdet deps ok')"
python -c "import cv2, boxmot, lap, pycocotools, motmetrics; print('tracking deps ok')"
```

The environment pins PyTorch 2.4.1, torchvision 0.19.1, torchaudio 2.4.1, and the PyTorch CUDA 12.1 runtime. `pip-constraints.txt` keeps pip-installed packages from replacing that Conda CUDA stack. Your NVIDIA driver must support CUDA 12.1, but the locally installed CUDA Toolkit does not need to match exactly because PyTorch ships the runtime libraries it uses.

After rebuilding the environment, reclaim package-cache space with:

```powershell
conda clean --all -y
python -m pip cache purge
```

## Dataset Preparation

Place CholecTrack20 files under:

```text
dataset/cholecTrack20/
dataset/annotations/
```

Convert the official CholecTrack20 JSON annotations into YOLO-style labels. EfficientDet uses these labels through the project dataset loader:

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

Default EfficientDet-D0 hyperparameters are defined in `configs/efficientdet_cholecTrack20.yaml`:

- `model`: `tf_efficientdet_d0`
- `epochs`: 100, tunable up to 300
- `batch`: 4
- `accumulation`: 2
- `imgsz`: 512
- `lr0`: 0.0002
- `lrf`: 0.01
- `optimizer`: AdamW
- `weight_decay`: 0.0001
- `warmup_epochs`: 3

Use presets to keep iteration fast and final runs accurate:

The `quick` preset intentionally uses a tiny training fraction so it can prove the EfficientDet pipeline works in minutes. Use `tune` or `final` for meaningful detector quality.

```powershell
# Quick: pipeline sanity check
python -m src.train train-detector --config configs/efficientdet_cholecTrack20.yaml --preset quick --epochs 1 --device 0
python -m src.train evaluate-detector --weights results/weights/efficientdet_best.pth --split val --half
python -m src.train paper-detection-eval --weights results/weights/efficientdet_best.pth --split val --half
python -m src.train benchmark-trackers --weights results/weights/efficientdet_best.pth --source dataset/cholecTrack20/Testing --preset smoke --trackers strongsort --device 0

# Tune: compare EfficientDet-D0 hyperparameters
python -m src.train train-detector --config configs/efficientdet_cholecTrack20.yaml --preset tune --epochs 100 --device 0 --name efficientdet_d0_tune_lr2e4
python -m src.train evaluate-detector --weights results/weights/efficientdet_best.pth --split val --half
python -m src.train paper-detection-eval --weights results/weights/efficientdet_best.pth --split val --half

# Overnight sweep: run EfficientDet-D0 hyperparameter candidates sequentially
python -m src.train sweep-detector --config configs/efficientdet_cholecTrack20.yaml --preset tune --epochs 50 --device 0 --half

# Final: reportable detector and tracker results
python -m src.train train-detector --config configs/efficientdet_cholecTrack20.yaml --preset final --epochs 300 --patience 30 --device 0 --name final_efficientdet_d0
python -m src.train evaluate-detector --weights results/weights/efficientdet_best.pth --split val
python -m src.train paper-detection-eval --weights results/weights/efficientdet_best.pth --split val
python -m src.train benchmark-trackers --weights results/weights/efficientdet_best.pth --source dataset/cholecTrack20/Testing --trackers strongsort --perspectives visibility intracorporeal intraoperative --device 0
```

EfficientDet weights and logs are written to `results/weights/` and `results/logs/`.
The shortcut checkpoint is `results/weights/efficientdet_best.pth`; each run also keeps its own checkpoint under `results/logs/<run_name>/weights/`.
The exported detector metrics are written to `results/logs/detection_metrics.csv`.
The CAMMA-style detector table row is written by `paper-detection-eval` to `results/logs/efficientdet_paper_detection_metrics.csv` and `.md`.
Every major train/eval/tracking command appends to `results/logs/experiment_log.csv`.
For faster overnight tuning, add `--eval-interval 3` to validate every third epoch instead of every epoch; final runs should use the default `1`.

## Tracking

Run tracking with the trained detector:

```bash
python -m src.train track --weights results/weights/efficientdet_best.pth --source dataset/cholecTrack20/Testing --tracker strongsort --perspective visibility --device 0
```

By default this writes tracking text outputs without rendering annotated videos, which keeps memory use lower on long surgical videos. EfficientDet checkpoints are fed directly into StrongSORT frame-by-frame; they do not go through BoxMOT's detector CLI.

For a quick pipeline smoke test, run the sampled benchmark preset:

```bash
python -m src.train benchmark-trackers --weights results/weights/efficientdet_best.pth --source dataset/cholecTrack20/Testing --preset smoke --trackers strongsort --device 0
```

The smoke preset runs `VID01` at `--vid-stride 25` on the `visibility` perspective. For EfficientDet checkpoints, use `strongsort`. Use explicit controls for custom fast checks:

```bash
python -m src.train benchmark-trackers --weights results/weights/efficientdet_best.pth --source dataset/cholecTrack20/Testing --videos VID01 VID06 --trackers strongsort --perspectives visibility --vid-stride 25 --jobs 1 --device 0
```

Benchmark all supported tracker modes across all CholecTrack20 tracking perspectives for final results:

```bash
python -m src.train benchmark-trackers --weights results/weights/efficientdet_best.pth --source dataset/cholecTrack20/Testing --trackers strongsort --perspectives visibility intracorporeal intraoperative --device 0
```

This writes:

- `results/logs/tracking_visibility.csv`
- `results/logs/tracking_intracorporeal.csv`
- `results/logs/tracking_intraoperative.csv`
- `results/logs/tracking_summary.csv`

Evaluate the generated MOT predictions against CholecTrack20 track IDs:

```bash
python -m src.train evaluate-tracking --trackers strongsort --perspectives visibility intracorporeal intraoperative --iou-threshold 0.5
```

This writes reportable tracking metrics:

- `results/logs/tracking_eval_summary.csv`
- `results/logs/tracking_eval_by_sequence.csv`
- `EA_STRONGSORT_TRACKING_EVALUATION.md`

The evaluator compares annotated CholecTrack20 frames only and uses class-aware IoU matching by default. Add `--class-agnostic` only as a diagnostic check.

## YOLO11 Baseline

YOLO11 was tested first and rejected as the active detector because its AP was insufficient for the project goal. The historical results are kept in:

- `YOLO11_HYPERPARAMETER_COMPARISON.md`
- `YOLO11_PAPER_STYLE_DETECTION_RESULTS.md`

Legacy YOLO commands are still available with `--detector-backend yolo`, but new experiments should use EfficientDet-D0.

## Baseline References

- StrongSORT upstream implementation: https://github.com/dyhBUPT/StrongSORT
- EfficientDet PyTorch implementation: https://github.com/rwightman/efficientdet-pytorch
- CholecTrack20 benchmark reference: https://github.com/CAMMA-public/cholectrack20

## GitHub Notes

Large assets are excluded by `.gitignore`: datasets, trained weights, model exports, logs, and environment-local files. Commit code, configuration, documentation, reports, and slides; keep datasets and generated outputs outside version control or use Git LFS only when explicitly needed.
