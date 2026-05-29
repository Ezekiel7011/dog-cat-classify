from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import PipelineConfig, load_config
from .evaluate import load_checkpoint
from .io import write_json
from .models import build_model
from .runtime import ensure_dir, resolve_device


def benchmark(config: PipelineConfig, checkpoint_path: str | Path, warmup: int = 10, runs: int = 50) -> dict[str, float | str]:
    inference = config.inference
    device = resolve_device(inference.device if inference else "auto")
    model = build_model(config.model.name, config.model.num_classes).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    batch_size = inference.batch_size if inference else 1
    sample = torch.randn(batch_size, 3, *config.data.image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(runs):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    latency_ms = elapsed / runs * 1000
    throughput = batch_size * runs / elapsed
    result = {
        "device": str(device),
        "batch_size": float(batch_size),
        "runs": float(runs),
        "latency_ms_per_batch": latency_ms,
        "throughput_images_per_second": throughput,
    }
    write_json(ensure_dir(config.artifacts.report_dir) / "benchmark.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AOI model inference")
    parser.add_argument("--config", default="configs/infer.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    result = benchmark(load_config(args.config), args.checkpoint, args.warmup, args.runs)
    print(
        "latency_ms_per_batch={latency_ms_per_batch:.3f} "
        "throughput_images_per_second={throughput_images_per_second:.3f}".format(**result)
    )


if __name__ == "__main__":
    main()
