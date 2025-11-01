"""
train.py — stable custom LF-YOLO training (Ultralytics ≥8.3)
"""

import torch
if not torch.cuda.is_available():
    raise SystemError("CUDA not detected! Make sure you're running inside the venv with the GPU-enabled torch build.")

from ultralytics import YOLO
from lf_yolo import LFYOLO_WeldDefect

DATA_PATH = "weld_defects/weld_defects.yaml"
# Full training configuration for RTX 4090 16GB
EPOCHS = 200           # run a full training
IMG_SIZE = 640         # standard YOLO size; stable and fast
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = -1        # AutoBatch: find the largest batch that fits
AMP = DEVICE == "cuda"

print(f"\nTraining on: {DEVICE.upper()} (AMP={AMP})")

# --- create your model ---
model = LFYOLO_WeldDefect(nc=7)
model.to(DEVICE)
print(f"[INFO] Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# Build YOLO training stack from a small checkpoint so Ultralytics sets up
# its Trainer/Loss correctly, then drop in our custom model. Using a .pt
# (not .yaml) ensures ckpt is populated and avoids rebuilding from cfg.
yolo = YOLO('yolov8n.pt')
yolo.model = model

# Ensure Ultralytics sees correct metadata
yolo.model.nc = model.nc
yolo.model.names = model.names
yolo.model.stride = model.stride
yolo.model.model[-1].nc = model.nc
yolo.model.model[-1].stride = model.stride

yolo.train(
    data=DATA_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=0,
    workers=0,          # Windows-friendly dataloader
    cos_lr=True,        # cosine LR schedule
    patience=50,        # early stop patience (max epochs still 200)
    save_period=10,     # save a checkpoint every 10 epochs
    name="lf_yolo_weld_training",
    project="runs/detect",
    amp=AMP,
    plots=True,
    val=True,
    exist_ok=True,
    verbose=True
)

print("\nTraining complete!  Check 'runs/detect/lf_yolo_weld_training/' for results.")
