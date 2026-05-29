param(
  [string]$Checkpoint = "artifacts/checkpoints/best.pt"
)

$env:PYTHONPATH = "src"
python -m aoi_inspection.evaluate --config configs/train.yaml --checkpoint $Checkpoint
