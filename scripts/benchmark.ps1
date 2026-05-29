param(
  [string]$Checkpoint = "artifacts/checkpoints/best.pt"
)

$env:PYTHONPATH = "src"
python -m aoi_inspection.benchmark --config configs/infer.yaml --checkpoint $Checkpoint
