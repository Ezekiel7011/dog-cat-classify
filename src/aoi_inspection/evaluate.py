from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import PipelineConfig, load_config
from .datasets import build_image_folder
from .io import write_json
from .metrics import ClassificationMetrics, compute_metrics
from .models import build_model
from .runtime import ensure_dir, resolve_device, set_seed


LOGGER = logging.getLogger(__name__)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return checkpoint


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total += int(labels.size(0))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, classes)
    return {"loss": total_loss / max(total, 1), "metrics": metrics}


def evaluate(config: PipelineConfig, checkpoint_path: str | Path) -> ClassificationMetrics:
    set_seed(config.seed)
    if config.data.val_dir is None:
        raise ValueError("Evaluation requires data.val_dir")

    device_name = config.training.device if config.training else "auto"
    device = resolve_device(device_name)
    dataset = build_image_folder(config.data.val_dir, config.data, training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size if config.training else 16,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    model = build_model(config.model.name, config.model.num_classes).to(device)
    load_checkpoint(model, checkpoint_path, device)
    result = evaluate_model(model, loader, nn.CrossEntropyLoss(), device, config.data.classes)
    metrics = result["metrics"]

    report_dir = ensure_dir(config.artifacts.report_dir)
    write_json(report_dir / "evaluation_metrics.json", metrics)
    LOGGER.info("Validation loss: %.4f", result["loss"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AOI inspection model")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = evaluate(load_config(args.config), args.checkpoint)
    print(f"accuracy={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f}")


if __name__ == "__main__":
    main()
