from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import PipelineConfig, load_config
from .evaluate import load_checkpoint
from .models import build_model
from .runtime import ensure_dir, resolve_device


def export_onnx(config: PipelineConfig, checkpoint_path: str | Path, output_path: str | Path | None = None) -> Path:
    inference = config.inference
    device = resolve_device(inference.device if inference else "cpu")
    model = build_model(config.model.name, config.model.num_classes).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    onnx_dir = ensure_dir(config.artifacts.onnx_dir)
    output = Path(output_path) if output_path else onnx_dir / f"{config.model.name}.onnx"
    dummy = torch.randn(1, 3, *config.data.image_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an AOI inspection model to ONNX")
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    output = export_onnx(load_config(args.config), args.checkpoint, args.output)
    print(f"onnx={output}")


if __name__ == "__main__":
    main()
