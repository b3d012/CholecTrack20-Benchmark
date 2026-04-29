"""Utilities for dataset validation, logging, metrics, and visualization."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml


def check_environment() -> dict[str, Any]:
    return {
        "python_torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file did not contain a mapping: {path}")
    return data


def validate_yolo_dataset(data_yaml: str | Path) -> list[str]:
    data = load_yaml(data_yaml)
    root = Path(data.get("path", "."))
    problems: list[str] = []

    for split in ("train", "val", "test"):
        split_value = data.get(split)
        if split_value is None:
            if split != "test":
                problems.append(f"Missing `{split}` entry in {data_yaml}")
            continue
        split_path = root / split_value
        if not split_path.exists():
            problems.append(f"Missing {split} path: {split_path}")

    names = data.get("names")
    if not names:
        problems.append("Missing class `names` in dataset YAML.")

    return problems


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_metric_csv(csv_path: str | Path, output_path: str | Path) -> None:
    frame = pd.read_csv(csv_path)
    numeric_columns = [
        column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_columns:
        raise ValueError(f"No numeric columns found in {csv_path}")

    plt.figure(figsize=(10, 6))
    for column in numeric_columns:
        if column.lower() not in {"epoch", "step"}:
            x = frame["epoch"] if "epoch" in frame.columns else frame.index
            plt.plot(x, frame[column], label=column)
    plt.xlabel("epoch" if "epoch" in frame.columns else "step")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

