"""YOLOv11 detector helpers and EfficientNetV2/ECA building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    try:
        import torch
        from torch import nn
    except ImportError:
        torch = None

        class _MissingNN:
            class Module:
                pass

        nn = _MissingNN()


class ECALayer(nn.Module):
    """Efficient Channel Attention for convolutional feature maps."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for ECALayer. Run `conda env create -f environment.yml`."
            )
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd.")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class EfficientNetV2ECA(nn.Module):
    """Feature extractor for ablation experiments and ReID embeddings."""

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        use_eca: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for EfficientNetV2ECA. "
                "Run `conda env create -f environment.yml`."
            )
        super().__init__()
        from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.use_eca = use_eca
        self.eca = ECALayer(1280) if use_eca else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.features(x)
        x = self.eca(x)
        x = self.pool(x)
        return self.embedding(x)


@dataclass(frozen=True)
class DetectorConfig:
    model: str = "yolo11n.pt"
    data: str = "dataset/cholecTrack20.yaml"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    project: str = "results/logs"
    name: str = "yolov11_cholectrack20"
    lr0: float = 0.01
    lrf: float = 0.01
    optimizer: str = "SGD"
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    weights_dir: str = "results/weights"


def load_yolov11(model: str = "yolo11n.pt") -> Any:
    """Load a YOLOv11 model through Ultralytics."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run `conda env create -f environment.yml`."
        ) from exc
    return YOLO(model)


def train_yolov11(config: DetectorConfig) -> Any:
    model = load_yolov11(config.model)
    result = model.train(
        data=config.data,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        project=config.project,
        name=config.name,
        lr0=config.lr0,
        lrf=config.lrf,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        exist_ok=True,
    )
    _copy_best_weights(config, result, model)
    return result


def _copy_best_weights(config: DetectorConfig, result: Any | None = None, model: Any | None = None) -> None:
    run_dirs: list[Path] = []
    for source in (result, getattr(model, "trainer", None)):
        save_dir = getattr(source, "save_dir", None)
        if save_dir:
            run_dirs.append(Path(save_dir) / "weights")

    run_dirs.extend(
        [
            Path(config.project) / config.name / "weights",
            Path("runs/detect") / config.project / config.name / "weights",
        ]
    )

    target_dir = Path(config.weights_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    import shutil

    copied = False
    for run_dir in dict.fromkeys(run_dirs):
        best = run_dir / "best.pt"
        last = run_dir / "last.pt"
        if best.exists():
            shutil.copy2(best, target_dir / "best.pt")
            copied = True
        if last.exists():
            shutil.copy2(last, target_dir / "last.pt")
            copied = True
        if copied:
            return

    searched = ", ".join(str(run_dir) for run_dir in run_dirs)
    raise FileNotFoundError(f"Could not find trained weights. Searched: {searched}")


def validate_yolov11(
    weights: str | Path,
    data: str | Path,
    device: str = "0",
    split: str = "val",
) -> Any:
    model = load_yolov11(str(weights))
    return model.val(data=str(data), device=device, split=split)
