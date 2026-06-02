"""EfficientDet-D0 detector training, evaluation, and inference helpers."""

from __future__ import annotations

import csv
import os
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .cholec_detection_eval import (
    CHALLENGES,
    CLASS_NAMES,
    _challenge_image_ids,
    _pct,
    _subset_gt,
    build_coco_gt,
    coco_ap,
    write_outputs,
)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class EfficientDetConfig:
    model: str = "tf_efficientdet_d0"
    data: str = "dataset/yolo_cholecTrack20/cholecTrack20.yaml"
    source: str = "dataset/cholecTrack20"
    epochs: int = 100
    imgsz: int = 512
    batch: int = 4
    accumulation: int = 1
    device: str = "0"
    project: str = "results/logs"
    name: str = "efficientdet_d0_cholectrack20"
    lr0: float = 0.0002
    lrf: float = 0.01
    optimizer: str = "AdamW"
    weight_decay: float = 0.0001
    warmup_epochs: int = 3
    weights_dir: str = "results/weights"
    workers: int = 4
    patience: int = 30
    fraction: float = 1.0
    plots: bool = False
    resume: bool = False
    amp: bool = True
    pretrained: bool = True
    num_classes: int = 7
    eval_interval: int = 1


def _device(value: str) -> torch.device:
    if value.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{value}")


def _load_data_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _split_image_dir(data_yaml: str | Path, split: str) -> Path:
    data = _load_data_yaml(data_yaml)
    root = Path(data["path"])
    return root / str(data[split])


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _letterbox_pil(image: Image.Image, imgsz: int) -> tuple[Image.Image, float, float, float]:
    width, height = image.size
    scale = min(imgsz / width, imgsz / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(resized, Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    pad_x = (imgsz - resized[0]) / 2.0
    pad_y = (imgsz - resized[1]) / 2.0
    canvas.paste(image, (int(round(pad_x)), int(round(pad_y))))
    return canvas, scale, pad_x, pad_y


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    classes: list[int] = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = [float(part) for part in line.split()]
        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height
        boxes.append([x1, y1, x2, y2])
        classes.append(int(cls) + 1)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


def _transform_boxes(
    boxes: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    imgsz: int,
    *,
    yxyx: bool,
) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    transformed = boxes.copy()
    transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_x
    transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_y
    transformed[:, [0, 2]] = np.clip(transformed[:, [0, 2]], 0, imgsz)
    transformed[:, [1, 3]] = np.clip(transformed[:, [1, 3]], 0, imgsz)
    keep = (transformed[:, 2] > transformed[:, 0]) & (transformed[:, 3] > transformed[:, 1])
    transformed = transformed[keep]
    if yxyx and transformed.size:
        transformed = transformed[:, [1, 0, 3, 2]]
    return transformed.astype(np.float32)


class CholecYoloDetectionDataset(Dataset):
    def __init__(self, data_yaml: str | Path, split: str, imgsz: int) -> None:
        self.data_yaml = Path(data_yaml)
        self.split = split
        self.imgsz = int(imgsz)
        image_dir = _split_image_dir(data_yaml, split)
        self.images = sorted(
            path for path in image_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.images:
            raise FileNotFoundError(f"No images found for split '{split}' in {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes, classes = _read_yolo_labels(_label_path_for_image(image_path), width, height)
        image, scale, pad_x, pad_y = _letterbox_pil(image, self.imgsz)
        boxes = _transform_boxes(boxes, scale, pad_x, pad_y, self.imgsz, yxyx=True)
        classes = classes[: boxes.shape[0]]
        if boxes.shape[0] == 0:
            boxes = np.zeros((1, 4), dtype=np.float32)
            classes = np.full((1,), -1, dtype=np.int64)

        target = {
            "bbox": torch.as_tensor(boxes, dtype=torch.float32),
            "cls": torch.as_tensor(classes, dtype=torch.int64),
            "img_size": torch.tensor([self.imgsz, self.imgsz], dtype=torch.float32),
            "img_scale": torch.tensor(1.0, dtype=torch.float32),
        }
        meta = {
            "path": image_path.as_posix(),
            "width": width,
            "height": height,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
        }
        return _image_to_tensor(image), target, meta


def _collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]]):
    images, targets, metas = zip(*batch)
    max_boxes = max(int(target["bbox"].shape[0]) for target in targets)
    padded_boxes = []
    padded_classes = []
    for target in targets:
        count = int(target["bbox"].shape[0])
        boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
        classes = torch.full((max_boxes,), -1, dtype=torch.int64)
        boxes[:count] = target["bbox"]
        classes[:count] = target["cls"]
        padded_boxes.append(boxes)
        padded_classes.append(classes)
    return (
        torch.stack(list(images)),
        {
            "bbox": torch.stack(padded_boxes),
            "cls": torch.stack(padded_classes),
            "img_size": torch.stack([target["img_size"] for target in targets]),
            "img_scale": torch.stack([target["img_scale"] for target in targets]),
        },
        list(metas),
    )


def _maybe_fraction(dataset: Dataset, fraction: float) -> Dataset:
    if fraction >= 1.0:
        return dataset
    size = max(1, int(len(dataset) * max(fraction, 0.0)))
    return Subset(dataset, list(range(size)))


def _loader(dataset: Dataset, batch: int, workers: int, shuffle: bool) -> DataLoader:
    workers = max(int(workers), 0)
    kwargs: dict[str, Any] = {}
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
        **kwargs,
    )


def create_efficientdet(config: EfficientDetConfig, *, bench_task: str, pretrained: bool | None = None) -> Any:
    try:
        from effdet import create_model
    except ImportError as exc:
        raise RuntimeError(
            "EfficientDet dependencies are missing. Install/update the environment with "
            "`python -m pip install effdet==0.4.1 timm==0.9.16 omegaconf==2.3.0`."
        ) from exc

    use_pretrained = config.pretrained if pretrained is None else pretrained
    cache_home = Path("results/logs/model_cache").resolve()
    cache_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_home / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(cache_home / "torch"))
    os.environ.setdefault("TIMM_CACHE_DIR", str(cache_home / "timm"))

    return create_model(
        config.model,
        bench_task=bench_task,
        num_classes=config.num_classes,
        pretrained=use_pretrained,
        pretrained_backbone=use_pretrained,
        bench_labeler=bench_task == "train",
        image_size=(config.imgsz, config.imgsz),
    )


def _save_checkpoint(path: Path, model: Any, config: EfficientDetConfig, epoch: int, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_owner = getattr(model, "model", model)
    torch.save(
        {
            "model": config.model,
            "num_classes": config.num_classes,
            "imgsz": config.imgsz,
            "epoch": epoch,
            "metrics": metrics,
            "state_dict": state_owner.state_dict(),
        },
        path,
    )


def _load_checkpoint(weights: str | Path, device: torch.device) -> tuple[Any, dict[str, Any], EfficientDetConfig]:
    checkpoint = torch.load(weights, map_location=device)
    cfg = EfficientDetConfig(
        model=checkpoint.get("model", "tf_efficientdet_d0"),
        imgsz=int(checkpoint.get("imgsz", 512)),
        num_classes=int(checkpoint.get("num_classes", 7)),
        pretrained=False,
    )
    model = create_efficientdet(cfg, bench_task="predict", pretrained=False)
    state_owner = getattr(model, "model", model)
    state_owner.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, cfg


def _optimizer(config: EfficientDetConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr0,
            momentum=0.937,
            weight_decay=config.weight_decay,
            nesterov=True,
        )
    return torch.optim.AdamW(model.parameters(), lr=config.lr0, weight_decay=config.weight_decay)


def _scheduler(config: EfficientDetConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_epochs = max(int(config.warmup_epochs), 0)
    cosine_epochs = max(int(config.epochs) - warmup_epochs, 1)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return max((epoch + 1) / warmup_epochs, 1e-3)
        progress = min(max((epoch - warmup_epochs) / cosine_epochs, 0.0), 1.0)
        return config.lrf + (1.0 - config.lrf) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_efficientdet(config: EfficientDetConfig) -> dict[str, Any]:
    device = _device(config.device)
    run_dir = Path(config.project) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_data = _maybe_fraction(CholecYoloDetectionDataset(config.data, "train", config.imgsz), config.fraction)
    val_data = CholecYoloDetectionDataset(config.data, "val", config.imgsz)
    train_loader = _loader(train_data, config.batch, config.workers, shuffle=True)

    model = create_efficientdet(config, bench_task="train").to(device)
    optimizer = _optimizer(config, model)
    scheduler = _scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    accumulation = max(int(config.accumulation), 1)

    best_map = -1.0
    bad_epochs = 0
    history_path = run_dir / "results.csv"
    if not history_path.exists():
        history_path.write_text(
            "epoch,train_loss,AP0.5,AP0.75,AP0.5:0.95,lr,train_time_sec,eval_time_sec,epoch_time_sec\n",
            encoding="utf-8",
        )

    for epoch in range(1, int(config.epochs) + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        batches = 0
        train_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, targets, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                losses = model(images, targets)
                loss = losses["loss"] if isinstance(losses, dict) else losses
                loss = loss / accumulation
            scaler.scale(loss).backward()
            if batch_idx % accumulation == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * accumulation
            batches += 1

        train_time = time.perf_counter() - train_started
        scheduler.step()
        train_loss = total_loss / max(batches, 1)
        eval_started = time.perf_counter()
        should_eval = epoch == 1 or epoch % max(int(config.eval_interval), 1) == 0
        if should_eval:
            metrics = evaluate_efficientdet(
                weights=None,
                data=config.data,
                source=config.source,
                split="val",
                device=config.device,
                imgsz=config.imgsz,
                batch=config.batch,
                half=config.amp,
                model=model,
            )
        else:
            metrics = {"AP0.5": "", "AP0.75": "", "AP0.5:0.95": "", "fps": ""}
        eval_time = time.perf_counter() - eval_started
        epoch_time = time.perf_counter() - epoch_started
        current_map = float(metrics.get("AP0.5:0.95", 0.0) or 0.0) if should_eval else best_map
        ap50 = metrics.get("AP0.5", "")
        ap75 = metrics.get("AP0.75", "")
        ap5095 = metrics.get("AP0.5:0.95", "")
        ap50_text = f"{float(ap50):.6f}" if ap50 != "" else ""
        ap75_text = f"{float(ap75):.6f}" if ap75 != "" else ""
        ap5095_text = f"{float(ap5095):.6f}" if ap5095 != "" else ""
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{train_loss:.6f},{ap50_text},{ap75_text},{ap5095_text},"
                f"{optimizer.param_groups[0]['lr']:.8f},"
                f"{train_time:.2f},{eval_time:.2f},{epoch_time:.2f}\n"
            )

        _save_checkpoint(weights_dir / "last.pth", model, config, epoch, metrics)
        if should_eval and current_map > best_map:
            best_map = current_map
            bad_epochs = 0
            _save_checkpoint(weights_dir / "best.pth", model, config, epoch, metrics)
            _copy_shortcuts(weights_dir / "best.pth", weights_dir / "last.pth", Path(config.weights_dir))
        elif should_eval:
            bad_epochs += 1

        ap50_print = f"{float(ap50):.4f}" if ap50 != "" else "skipped"
        map_print = f"{current_map:.4f}" if should_eval else "skipped"
        print(
            f"epoch {epoch}: loss={train_loss:.4f} AP0.5={ap50_print} "
            f"AP0.5:0.95={map_print} best={best_map:.4f} "
            f"time={epoch_time / 60.0:.1f}m train={train_time / 60.0:.1f}m eval={eval_time / 60.0:.1f}m"
        )
        if config.patience and bad_epochs >= config.patience:
            print(f"Early stopping after {bad_epochs} epochs without AP0.5:0.95 improvement.")
            break

    return {"save_dir": str(run_dir), "best_map": best_map, "weights": str(weights_dir / "best.pth")}


def _copy_shortcuts(best: Path, last: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, target_dir / "efficientdet_best.pth")
        shutil.copy2(best, target_dir / "best.pth")
    if last.exists():
        shutil.copy2(last, target_dir / "efficientdet_last.pth")
        shutil.copy2(last, target_dir / "last.pth")


@torch.inference_mode()
def predict_efficientdet_images(
    weights: str | Path | None,
    image_paths: Iterable[str | Path],
    *,
    device: str = "0",
    imgsz: int | None = None,
    batch: int = 4,
    half: bool = False,
    model: Any | None = None,
) -> tuple[list[dict[str, Any]], float]:
    torch_device = _device(device)
    if model is None:
        if weights is None:
            raise ValueError("weights are required when model is not provided")
        model, _, checkpoint_config = _load_checkpoint(weights, torch_device)
        imgsz = imgsz or checkpoint_config.imgsz
    else:
        if model.__class__.__name__ == "DetBenchTrain":
            temp_config = EfficientDetConfig(imgsz=int(imgsz or 512), pretrained=False)
            predict_model = create_efficientdet(temp_config, bench_task="predict", pretrained=False)
            predict_model.model.load_state_dict(model.model.state_dict(), strict=True)
            model = predict_model
        model = model.to(torch_device).eval()
        imgsz = imgsz or int(getattr(getattr(model, "config", None), "image_size", 512) or 512)

    paths = [Path(path) for path in image_paths]
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(paths), max(int(batch), 1)):
        chunk = paths[start : start + max(int(batch), 1)]
        tensors = []
        metas = []
        for path in chunk:
            original = Image.open(path).convert("RGB")
            width, height = original.size
            image, scale, pad_x, pad_y = _letterbox_pil(original, int(imgsz))
            tensors.append(_image_to_tensor(image))
            metas.append((path.as_posix(), width, height, scale, pad_x, pad_y))
        images = torch.stack(tensors).to(torch_device)
        if half and torch_device.type == "cuda":
            images = images.half()
            model = model.half()
        detections = model(images)
        detections = detections.detach().float().cpu().numpy()
        for dets, (path, width, height, scale, pad_x, pad_y) in zip(detections, metas):
            for det in dets:
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, score, cls_id = [float(value) for value in det[:6]]
                if score <= 0:
                    continue
                x1 = (x1 - pad_x) / scale
                x2 = (x2 - pad_x) / scale
                y1 = (y1 - pad_y) / scale
                y2 = (y2 - pad_y) / scale
                x1 = min(max(x1, 0.0), width)
                x2 = min(max(x2, 0.0), width)
                y1 = min(max(y1, 0.0), height)
                y2 = min(max(y2, 0.0), height)
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append(
                    {
                        "path": path,
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                        "category_id": max(int(cls_id), 1),
                        "class_id": max(int(cls_id), 1) - 1,
                    }
                )
    elapsed = max(time.perf_counter() - started, 1e-9)
    return predictions, len(paths) / elapsed


def _gt_image_id_by_path(gt: dict[str, Any]) -> dict[str, int]:
    return {Path(image["file_name"]).as_posix(): int(image["id"]) for image in gt["images"]}


class EfficientDetPredictor:
    def __init__(
        self,
        weights: str | Path,
        *,
        device: str = "0",
        imgsz: int | None = None,
        half: bool = False,
    ) -> None:
        self.device = _device(device)
        self.model, _, checkpoint_config = _load_checkpoint(weights, self.device)
        self.imgsz = int(imgsz or checkpoint_config.imgsz)
        self.half = bool(half and self.device.type == "cuda")
        if self.half:
            self.model = self.model.half()

    @torch.inference_mode()
    def predict_pil(self, image: Image.Image, *, conf: float = 0.25) -> np.ndarray:
        image = image.convert("RGB")
        width, height = image.size
        letterboxed, scale, pad_x, pad_y = _letterbox_pil(image, self.imgsz)
        tensor = _image_to_tensor(letterboxed).unsqueeze(0).to(self.device)
        if self.half:
            tensor = tensor.half()
        detections = self.model(tensor).detach().float().cpu().numpy()[0]
        rows: list[list[float]] = []
        for det in detections:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, score, cls_id = [float(value) for value in det[:6]]
            if score < conf:
                continue
            x1 = (x1 - pad_x) / scale
            x2 = (x2 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            y2 = (y2 - pad_y) / scale
            x1 = min(max(x1, 0.0), width)
            x2 = min(max(x2, 0.0), width)
            y1 = min(max(y1, 0.0), height)
            y2 = min(max(y2, 0.0), height)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([x1, y1, x2, y2, score, max(int(cls_id), 1) - 1])
        if not rows:
            return np.empty((0, 6), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def predict_frame(self, frame_bgr: np.ndarray, *, conf: float = 0.25) -> np.ndarray:
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        return self.predict_pil(image, conf=conf)


def evaluate_efficientdet(
    weights: str | Path | None,
    data: str | Path,
    source: str | Path,
    split: str = "val",
    device: str = "0",
    imgsz: int | None = None,
    batch: int = 4,
    half: bool = False,
    model: Any | None = None,
) -> dict[str, float]:
    gt = build_coco_gt(Path(source), Path(_load_data_yaml(data)["path"]), split)
    image_id_by_path = _gt_image_id_by_path(gt)
    predictions, fps = predict_efficientdet_images(
        weights,
        image_id_by_path.keys(),
        device=device,
        imgsz=imgsz,
        batch=batch,
        half=half,
        model=model,
    )
    coco_predictions = []
    for prediction in predictions:
        image_id = image_id_by_path.get(Path(prediction["path"]).as_posix())
        if image_id is None:
            continue
        coco_predictions.append(
            {
                "image_id": image_id,
                "category_id": int(prediction["category_id"]),
                "bbox": prediction["bbox"],
                "score": float(prediction["score"]),
            }
        )
    overall = coco_ap(gt, coco_predictions)
    return {
        "AP0.5": float(overall["AP50"]),
        "AP0.75": float(overall["AP75"]),
        "AP0.5:0.95": float(overall["AP"]),
        "fps": float(fps),
    }


def cholec_evaluate_efficientdet(
    weights: str | Path,
    data: str | Path,
    source: str | Path,
    split: str,
    device: str,
    imgsz: int | None,
    batch: int,
    half: bool,
    model_name: str,
    output: str | Path,
) -> dict[str, str]:
    gt = build_coco_gt(Path(source), Path(_load_data_yaml(data)["path"]), split)
    image_id_by_path = _gt_image_id_by_path(gt)
    predictions, fps = predict_efficientdet_images(
        weights,
        image_id_by_path.keys(),
        device=device,
        imgsz=imgsz,
        batch=batch,
        half=half,
    )
    coco_predictions = [
        {
            "image_id": image_id_by_path[Path(prediction["path"]).as_posix()],
            "category_id": int(prediction["category_id"]),
            "bbox": prediction["bbox"],
            "score": float(prediction["score"]),
        }
        for prediction in predictions
        if Path(prediction["path"]).as_posix() in image_id_by_path
    ]
    overall = coco_ap(gt, coco_predictions)
    row: dict[str, str] = {
        "Detection model": model_name,
        "AP0.5": _pct(overall["AP50"]),
        "AP0.75": _pct(overall["AP75"]),
        "AP0.5:0.95": _pct(overall["AP"]),
    }
    for idx, class_name in enumerate(CLASS_NAMES, start=1):
        row[class_name] = _pct(coco_ap(gt, coco_predictions, cat_id=idx)["AP50"])
    for challenge_name, field in CHALLENGES.items():
        challenge_gt = _subset_gt(gt, _challenge_image_ids(gt, field))
        row[challenge_name] = _pct(coco_ap(challenge_gt, coco_predictions)["AP50"])
    row["FPS"] = f"{fps:.1f}"
    write_outputs(row, Path(output))
    return row
