# CholecTrack20 Benchmark: YOLO11, EfficientDet-D0, and EA-StrongSORT

This repository is a reproducible benchmark of surgical tool detection and multi-object tracking on CholecTrack20. It compares YOLO11 and EfficientDet-D0 as detector baselines, then uses the stronger detector output with EA-StrongSORT for tracking evaluation.

The project is organized for public sharing as a clean benchmark reference:

- source code, configs, and small benchmark summaries are tracked in Git
- datasets, checkpoints, generated logs, and large exports stay out of version control
- results are presented as a benchmark comparison, not as a dataset contribution
- the reported numbers were produced under the compute, time, and parameter-sweep limits available during the project

## Highlights

- Two detector baselines: YOLO11 and EfficientDet-D0
- CholecTrack20-specific detector evaluation with per-class and challenge metrics
- StrongSORT tracking benchmark across visibility, intracorporeal, and intraoperative perspectives
- Compact result tables stored under `benchmarks/` for easy citation and review

## Skills Demonstrated

- PyTorch-based model training and evaluation
- Ultralytics YOLO integration and detector benchmarking
- EfficientDet implementation and COCO-style metric reporting
- Multi-object tracking with StrongSORT and MOT evaluation
- Experiment management, result logging, and reproducible workflow design
- Clear technical documentation for public-facing portfolio use

## Benchmark Snapshot

The table below reflects the best-known runs recorded in this repo.

| Area | Model / Run | AP0.5 | AP0.75 | AP0.5:0.95 | FPS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Detection | EfficientDet-D0-640-AdamW-lr2e4 | 44.4 | 31.2 | 27.9 | 20.9 | Best EfficientDet detector run recorded here |
| Detection | YOLO11-img768-SGD | 40.3 | 29.6 | 26.6 | 62.9 | Historical baseline, kept for comparison |

Tracking evaluation with StrongSORT on annotated CholecTrack20 frames:

| Perspective | MOTA | IDF1 | Precision | Recall | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| visibility | 20.974 | 24.172 | 68.692 | 58.952 | Class-aware IoU >= 0.5 |
| intracorporeal | 19.641 | 12.158 | 68.692 | 58.952 | Class-aware IoU >= 0.5 |
| intraoperative | 19.314 | 6.298 | 68.692 | 58.952 | Class-aware IoU >= 0.5 |

## Scope And Limitations

This benchmark should be read as a careful, real-world project result rather than a perfectly exhaustive study.

- Training time was bounded by the available project schedule.
- Model search was limited to the detector families and parameter ranges that were practical to run end-to-end.
- The reported results are therefore the strongest benchmark numbers reached within those constraints, not an unconstrained global optimum.
- That makes the repo useful as a reproducible engineering benchmark, which is often exactly the kind of work recruiters want to see: clear problem framing, iteration, tradeoff management, and honest reporting.

## Repository Layout

```text
DeepLearningProject/
|-- benchmarks/
|-- configs/
|-- src/
|-- dataset/        # local only, ignored by Git
|-- results/        # local outputs, ignored by Git
`-- runs/           # local detector/tracking outputs, ignored by Git
```

## Setup

This project assumes a Conda-based Python environment.

```powershell
conda env create -f environment.yml
conda activate deep_learning_project
python -m src.train check-env
```

Recommended smoke checks:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip check
python -c "import cv2, boxmot, pycocotools, motmetrics; print('tracking deps ok')"
```

## Data Preparation

Place the CholecTrack20 dataset and annotations under:

```text
dataset/cholecTrack20/
dataset/annotations/
```

Prepare the YOLO-style labels used by the detector pipeline:

```powershell
python -m src.train prepare-data --source dataset/cholecTrack20 --out dataset/yolo_cholecTrack20
python -m src.train validate-data --data dataset/yolo_cholecTrack20/cholecTrack20.yaml
```

## Running The Benchmark

EfficientDet-D0 detector training and evaluation:

```powershell
python -m src.train train-detector --config configs/efficientdet_cholecTrack20.yaml --preset final --epochs 300 --patience 30 --device 0 --name final_efficientdet_d0
python -m src.train evaluate-detector --weights results/weights/efficientdet_best.pth --split val
python -m src.train cholec-detection-eval --weights results/weights/efficientdet_best.pth --split val --half
```

YOLO11 baseline training and evaluation:

```powershell
python -m src.train train-detector --config configs/yolov11_cholecTrack20.yaml --detector-backend yolo --preset final --device 0 --name final_yolo11
python -m src.train evaluate-detector --weights results/weights/best.pt --split val
python -m src.train cholec-detection-eval --weights results/weights/best.pt --split val --model-name YOLO11
```

Tracking benchmark and MOT evaluation:

```powershell
python -m src.train benchmark-trackers --weights results/weights/efficientdet_best.pth --source dataset/cholecTrack20/Testing --trackers strongsort --perspectives visibility intracorporeal intraoperative --device 0
python -m src.train evaluate-tracking --trackers strongsort --perspectives visibility intracorporeal intraoperative --iou-threshold 0.5
```

## Benchmark Files

The tracked benchmark summaries live under `benchmarks/`:

- `benchmarks/detection/efficientdet_detection_results.csv`
- `benchmarks/detection/efficientdet_hyperparameter_comparison.csv`
- `benchmarks/detection/yolo11_detection_results.md`
- `benchmarks/detection/yolo11_hyperparameter_comparison.md`
- `benchmarks/tracking/strongsort_tracking_results.csv`
- `benchmarks/tracking/strongsort_tracking_evaluation.csv`
- `benchmarks/tracking/strongsort_tracking_evaluation_class_agnostic.csv`
- `benchmarks/tracking/strongsort_tracking_per_video.csv`

## Reproducibility Notes

- This repository does not include the dataset, checkpoints, or generated reports.
- Benchmark numbers here are local experiment results and should be described as such in CVs, papers, or portfolios.
- When presenting this work, emphasize the engineering process: data preparation, detector comparison, tracking integration, evaluation, and careful tradeoff selection under compute/time constraints.
- The CholecTrack20 benchmark files are intentionally small so the repo stays readable and shareable.

## Citation

If you reference this repository in a paper, report, portfolio, or GitHub README, use the included `CITATION.cff` file or cite the project title directly:

> CholecTrack20 Benchmark: YOLO11, EfficientDet-D0, and EA-StrongSORT

## References

- CholecTrack20 benchmark: https://github.com/CAMMA-public/cholectrack20
- StrongSORT: https://github.com/dyhBUPT/StrongSORT
- EfficientDet PyTorch: https://github.com/rwightman/efficientdet-pytorch
