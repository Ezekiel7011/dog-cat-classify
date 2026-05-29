from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    classes: list[str]
    image_size: tuple[int, int]
    train_dir: Path | None = None
    val_dir: Path | None = None
    num_workers: int = 2


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 3
    device: str = "auto"


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int = 1
    device: str = "auto"


@dataclass(frozen=True)
class ArtifactConfig:
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    report_dir: Path = Path("artifacts/reports")
    onnx_dir: Path = Path("artifacts/onnx")


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    model: ModelConfig
    artifacts: ArtifactConfig
    training: TrainingConfig | None = None
    inference: InferenceConfig | None = None
    seed: int = 42
    name: str = "industrial-aoi-vision-pipeline"


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> PipelineConfig:
    project = raw.get("project", {})
    data = raw.get("data", {})
    model = raw.get("model", {})
    artifacts = raw.get("artifacts", {})

    classes = _required_list(data, "classes")
    image_size = tuple(_required_list(data, "image_size"))
    if len(image_size) != 2:
        raise ValueError("data.image_size must contain [height, width]")

    model_classes = int(model.get("num_classes", len(classes)))
    if model_classes != len(classes):
        raise ValueError("model.num_classes must match the number of data.classes")

    training = None
    if "training" in raw:
        train = raw["training"]
        training = TrainingConfig(
            batch_size=int(train.get("batch_size", 16)),
            epochs=int(train.get("epochs", 20)),
            learning_rate=float(train.get("learning_rate", 0.001)),
            weight_decay=float(train.get("weight_decay", 0.0001)),
            early_stopping_patience=int(train.get("early_stopping_patience", 3)),
            device=str(train.get("device", "auto")),
        )

    inference = None
    if "inference" in raw:
        infer = raw["inference"]
        inference = InferenceConfig(
            batch_size=int(infer.get("batch_size", 1)),
            device=str(infer.get("device", "auto")),
        )

    return PipelineConfig(
        name=str(project.get("name", "industrial-aoi-vision-pipeline")),
        seed=int(project.get("seed", 42)),
        data=DataConfig(
            classes=[str(item) for item in classes],
            image_size=(int(image_size[0]), int(image_size[1])),
            train_dir=_optional_path(data.get("train_dir")),
            val_dir=_optional_path(data.get("val_dir")),
            num_workers=int(data.get("num_workers", 2)),
        ),
        model=ModelConfig(
            name=str(model.get("name", "coatnet_0")),
            num_classes=model_classes,
        ),
        training=training,
        inference=inference,
        artifacts=ArtifactConfig(
            checkpoint_dir=Path(artifacts.get("checkpoint_dir", "artifacts/checkpoints")),
            report_dir=Path(artifacts.get("report_dir", "artifacts/reports")),
            onnx_dir=Path(artifacts.get("onnx_dir", "artifacts/onnx")),
        ),
    )


def _required_list(raw: dict[str, Any], key: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"data.{key} must be a non-empty list")
    return value


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))
