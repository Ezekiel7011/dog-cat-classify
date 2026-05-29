# Industrial AOI Vision Inspection Pipeline

Reference implementation for a production-oriented automated optical inspection
(AOI) workflow. The project uses a small public image-classification sample so the
repository can run end to end, while the structure mirrors the workflow used in
industrial vision systems: configurable training, metric-driven evaluation,
batch inference, ONNX export, and latency benchmarking.

This repository is intentionally positioned as a vision-system engineering
project, not just a model-training demo.

## Engineering Goals

- Build a reproducible inspection pipeline from dataset loading to deployment
  artifact export.
- Track metrics that matter in AOI: precision, recall, false call rate, miss
  rate, confusion matrix, latency, throughput, and model size.
- Keep the runtime configurable for different production lines, camera
  resolutions, classes, and hardware targets.
- Provide clean command-line entry points for training, validation, batch
  inference, ONNX export, and performance benchmarking.

## Architecture

```text
configs/                  Runtime configuration
docs/                     System design, model card, benchmark template
src/aoi_inspection/       Production-style Python package
  config.py               Typed config loading and validation
  datasets.py             ImageFolder data pipeline
  metrics.py              AOI-oriented classification metrics
  train.py                Training loop with checkpointing
  evaluate.py             Validation and report generation
  infer.py                Single-image and folder inference
  export_onnx.py          Deployment artifact export
  benchmark.py            Latency and throughput benchmark
  models/coatnet.py       CoAtNet model factory wrapper
scripts/                  Windows PowerShell examples
tests/                    Lightweight tests for config and metrics
```

## Quick Start

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train a model:

```powershell
python -m aoi_inspection.train --config configs/train.yaml
```

Evaluate a checkpoint:

```powershell
python -m aoi_inspection.evaluate --config configs/train.yaml --checkpoint artifacts/checkpoints/best.pt
```

Run batch inference:

```powershell
python -m aoi_inspection.infer --config configs/infer.yaml --input val --checkpoint artifacts/checkpoints/best.pt
```

Export ONNX:

```powershell
python -m aoi_inspection.export_onnx --config configs/infer.yaml --checkpoint artifacts/checkpoints/best.pt
```

Benchmark inference:

```powershell
python -m aoi_inspection.benchmark --config configs/infer.yaml --checkpoint artifacts/checkpoints/best.pt
```

For local development without package installation, set:

```powershell
$env:PYTHONPATH = "src"
```

## AOI Metrics

The evaluation report includes:

- accuracy
- macro precision / recall / F1
- per-class precision / recall / F1
- false call rate by class
- miss rate by class
- confusion matrix

In semiconductor AOI, accuracy alone can hide production risk. False calls
increase review cost and reduce line efficiency; misses create quality escapes.
This project therefore reports both rates explicitly.

## Dataset Note

The included `train/` and `val/` folders are tiny public sample images for
workflow validation only. For a real AOI project, replace them with an
ImageFolder-style dataset:

```text
dataset/
  train/
    pass/
    defect/
  val/
    pass/
    defect/
```

Large production images, customer data, checkpoints, and generated reports
should stay outside Git. Use `artifacts/` for local outputs.

## Production Hardening Roadmap

- Add 8K image tiling and region-level defect aggregation.
- Add ONNX Runtime parity tests against PyTorch output.
- Add threshold calibration using false call and miss-rate targets.
- Add production-line data drift summaries.
- Add CI checks for linting, unit tests, and smoke inference.

## Documents

- [System Design](docs/system_design.md)
- [Model Card](docs/model_card.md)
- [Benchmark Report Template](docs/benchmark_report.md)
