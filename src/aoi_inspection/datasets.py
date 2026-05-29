from __future__ import annotations

from pathlib import Path

import torch
from torchvision import datasets, transforms

from .config import DataConfig


def build_transforms(image_size: tuple[int, int], training: bool) -> transforms.Compose:
    steps: list[object] = [transforms.Resize(image_size)]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, padding=10),
            ]
        )
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def build_image_folder(path: Path, data: DataConfig, training: bool) -> datasets.ImageFolder:
    dataset = datasets.ImageFolder(path, transform=build_transforms(data.image_size, training))
    if dataset.classes != data.classes:
        raise ValueError(
            f"Dataset classes {dataset.classes} do not match config classes {data.classes}"
        )
    return dataset


def build_loaders(data: DataConfig, batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if data.train_dir is None or data.val_dir is None:
        raise ValueError("Training requires data.train_dir and data.val_dir")

    train_dataset = build_image_folder(data.train_dir, data, training=True)
    val_dataset = build_image_folder(data.val_dir, data, training=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data.num_workers,
    )
    return train_loader, val_loader
