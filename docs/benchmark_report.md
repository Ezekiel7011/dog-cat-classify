# Benchmark Report Template

## Hardware

- CPU:
- GPU:
- RAM:
- Storage:
- Runtime:

## Model

- Architecture:
- Checkpoint:
- Input size:
- Batch size:

## Results

| Metric | Value |
| --- | ---: |
| Latency per batch (ms) | |
| Throughput (images/sec) | |
| Model size (MB) | |
| Accuracy | |
| Macro F1 | |
| False call rate | |
| Miss rate | |

## Notes

- Record whether benchmark data came from PyTorch, ONNX Runtime, or production
  SDK.
- Compare results against the tact-time budget for the target station.
- Attach representative false call and miss cases for review.
