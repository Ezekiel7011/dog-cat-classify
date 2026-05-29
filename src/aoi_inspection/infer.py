from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import PipelineConfig, load_config
from .datasets import build_transforms
from .evaluate import load_checkpoint
from .io import write_json
from .models import build_model
from .runtime import ensure_dir, resolve_device, set_seed


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)


def predict(config: PipelineConfig, checkpoint_path: str | Path, input_path: str | Path) -> list[dict[str, object]]:
    set_seed(config.seed)
    inference = config.inference
    device = resolve_device(inference.device if inference else "auto")
    model = build_model(config.model.name, config.model.num_classes).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    transform = build_transforms(config.data.image_size, training=False)
    results: list[dict[str, object]] = []

    with torch.no_grad():
        for image_path in iter_images(Path(input_path)):
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            logits = model(tensor)
            probability = torch.softmax(logits, dim=1)[0]
            score, index = probability.max(dim=0)
            results.append(
                {
                    "path": str(image_path),
                    "prediction": config.data.classes[int(index.item())],
                    "score": float(score.item()),
                }
            )

    report_dir = ensure_dir(config.artifacts.report_dir)
    write_json(report_dir / "inference_results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AOI image inference")
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    results = predict(load_config(args.config), args.checkpoint, args.input)
    for row in results:
        print(f"{row['path']}\t{row['prediction']}\t{row['score']:.4f}")


if __name__ == "__main__":
    main()
