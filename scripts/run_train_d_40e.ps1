$ErrorActionPreference = "Stop"
Set-Location "C:\Users\Lucas\Downloads\POEM"
& "C:\Users\Lucas\AppData\Local\Programs\Python\Python314\python.exe" -u train.py `
  --model_type D `
  --data_dir Beautiful-Motifs-CC-BY-NC-SA `
  --epochs 40 `
  --batch_size 32 `
  --output_dir checkpoints `
  --log_interval 50 `
  --val_interval 2000 `
  --checkpoint_interval_steps 5000 `
  --token_cache cache\poem-short-token-cache.pt `
  *> logs\poem-d-40e.log
