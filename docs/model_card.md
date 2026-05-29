# Model Card

## Model

- Architecture: CoAtNet
- Task: image-level inspection classification
- Input: RGB image resized to the configured image size
- Output: class logits

## Intended Use

This project is a reference pipeline for AOI inspection engineering. The sample
dataset is intentionally small and should only be used for smoke testing. Real
production usage requires representative images from the target camera,
lighting, lens, material, and production recipe.

## Key Metrics

- Accuracy
- Macro precision, recall, and F1
- Per-class precision, recall, and F1
- False call rate
- Miss rate
- Latency and throughput

## Limitations

- The sample data is not representative of semiconductor AOI.
- Image resizing may remove small defects in high-resolution inspection images.
- Classification alone does not provide defect localization.
- Deployment parity should be verified after ONNX export.

## Recommended Validation

- Validate with lot-level splits to prevent leakage.
- Review false calls and misses separately with process engineers.
- Benchmark on the target IPC or GPU, not only on a development machine.
- Track metrics by product recipe and camera station.
