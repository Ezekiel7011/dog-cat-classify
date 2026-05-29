from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .config import PipelineConfig, load_config
from .datasets import build_loaders
from .evaluate import evaluate_model
from .io import write_json
from .models import build_model
from .runtime import ensure_dir, resolve_device, set_seed


LOGGER = logging.getLogger(__name__)


def train(config: PipelineConfig) -> Path:
    if config.training is None:
        raise ValueError("Training config is required")

    set_seed(config.seed)
    device = resolve_device(config.training.device)
    checkpoint_dir = ensure_dir(config.artifacts.checkpoint_dir)
    report_dir = ensure_dir(config.artifacts.report_dir)

    train_loader, val_loader = build_loaders(config.data, config.training.batch_size)
    model = build_model(config.model.name, config.model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_f1 = -1.0
    stale_epochs = 0
    best_checkpoint = checkpoint_dir / "best.pt"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += int(labels.size(0))

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)
        val_metrics = evaluate_model(model, val_loader, criterion, device, config.data.classes)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["metrics"].accuracy,
            "val_macro_f1": val_metrics["metrics"].macro_f1,
        }
        history.append(row)
        LOGGER.info("Epoch %s metrics: %s", epoch, row)

        if val_metrics["metrics"].macro_f1 > best_f1:
            best_f1 = val_metrics["metrics"].macro_f1
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": config.data.classes,
                    "model_name": config.model.name,
                    "num_classes": config.model.num_classes,
                    "image_size": config.data.image_size,
                },
                best_checkpoint,
            )
        else:
            stale_epochs += 1

        if stale_epochs >= config.training.early_stopping_patience:
            LOGGER.info("Early stopping after %s stale epochs", stale_epochs)
            break

    write_json(report_dir / "training_history.json", history)
    return best_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AOI inspection model")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint = train(load_config(args.config))
    print(f"best_checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
