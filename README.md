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
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').info()"
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

Use presets to keep iteration fast and final runs accurate:

```powershell
# Quick: pipeline sanity check
python -m src.train train-detector --config configs/yolov11_cholecTrack20.yaml --preset quick --device 0
python -m src.train evaluate-detector --weights results/weights/best.pt --split val --half
python -m src.train benchmark-trackers --weights results/weights/best.pt --source dataset/cholecTrack20/Testing --preset smoke --device 0

# Tune: compare YOLO hyperparameters
python -m src.train train-detector --config configs/yolov11_cholecTrack20.yaml --preset tune --epochs 100 --device 0
python -m src.train evaluate-detector --weights results/weights/best.pt --split val --half

# Overnight sweep: run several YOLO hyperparameter candidates sequentially
python -m src.train sweep-detector --config configs/yolov11_cholecTrack20.yaml --preset tune --epochs 50 --device 0 --half

# Final: reportable detector and tracker results
python -m src.train train-detector --config configs/yolov11_cholecTrack20.yaml --preset final --epochs 300 --device 0
python -m src.train evaluate-detector --weights results/weights/best.pt --split val
python -m src.train benchmark-trackers --weights results/weights/best.pt --source dataset/cholecTrack20/Testing --trackers botsort bytetrack ocsort strongsort --perspectives visibility intracorporeal intraoperative --device 0
```

Weights and logs are written to `results/weights/` and `results/logs/`.
The exported detector metrics are `Precision`, `Recall`, `mAP@0.5`, and `mAP@0.5:0.95` in `results/logs/detection_metrics.csv`.
The detector sweep command adds a timestamp tag automatically, writes a summary like `results/logs/detector_sweep_YYYYMMDD_HHMMSS.csv`, and keeps each run in its own folder under `results/logs/<tag>_<sweep_name>/`.

## Tracking

Run tracking with the trained detector:

```bash
python -m src.train track --weights results/weights/best.pt --source dataset/cholecTrack20/Testing --tracker strongsort --perspective visibility --device 0
```

By default this writes tracking text outputs without rendering annotated videos, which keeps memory use lower on long surgical videos. Add `--save-video` when running a short single-video track if you also need rendered video output.

For a quick pipeline smoke test, run the sampled benchmark preset:

```bash
python -m src.train benchmark-trackers --weights results/weights/best.pt --source dataset/cholecTrack20/Testing --preset smoke --device 0
```

The smoke preset runs `VID01` at `--vid-stride 25` with `botsort` and `bytetrack` on the `visibility` perspective. Use explicit controls for custom fast checks:

```bash
python -m src.train benchmark-trackers --weights results/weights/best.pt --source dataset/cholecTrack20/Testing --videos VID01 VID06 --trackers botsort --perspectives visibility --vid-stride 25 --jobs 2 --device 0
```

Benchmark all supported tracker modes across all CholecTrack20 tracking perspectives for final results:

```bash
python -m src.train benchmark-trackers --weights results/weights/best.pt --source dataset/cholecTrack20/Testing --trackers botsort bytetrack ocsort strongsort  --perspectives visibility intracorporeal intraoperative --device 0
```

This writes:

- `results/logs/tracking_visibility.csv`
- `results/logs/tracking_intracorporeal.csv`
- `results/logs/tracking_intraoperative.csv`
- `results/logs/tracking_summary.csv`

## Ablation Study

```bash
python -m src.train ablate --data dataset/yolo_cholecTrack20/cholecTrack20.yaml --device 0
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
