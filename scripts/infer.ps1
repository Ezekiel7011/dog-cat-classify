param(
  [string]$Checkpoint = "artifacts/checkpoints/best.pt",
  [string]$InputPath = "val"
)

$env:PYTHONPATH = "src"
python -m aoi_inspection.infer --config configs/infer.yaml --checkpoint $Checkpoint --input $InputPath
