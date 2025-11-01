"""
train.py — stable custom LF-YOLO training (Ultralytics ≥8.3)
"""

import torch
if not torch.cuda.is_available():
    raise SystemError("CUDA not detected! Make sure you're running inside the venv with the GPU-enabled torch build.")

from ultralytics import YOLO
from lf_yolo import LFYOLO_WeldDefect

DATA_PATH = "weld_defects/weld_defects.yaml"
EPOCHS = 2
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 if DEVICE == "cuda" else 4
AMP = DEVICE == "cuda"

print(f"\nTraining on: {DEVICE.upper()} (AMP={AMP})")

# --- create your model ---
model = LFYOLO_WeldDefect(nc=7)
model.to(DEVICE)
print(f"[INFO] Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# Ultralytics' YOLO() constructor often expects a weights path. Passing a
# raw PyTorch module may trigger a FileNotFoundError (it treats the object
# as a path string). To be robust, try to wrap the instance and fall back
# to creating a YOLO wrapper from a small official checkpoint and then
# replace its internal model with our custom model.
try:
    yolo = YOLO(model)
except FileNotFoundError:
    print('[INFO] YOLO() rejected the model instance; creating wrapper from yolov8n.pt and replacing its model')
    yolo = YOLO('yolov8n.pt')  # will download if missing
    # Replace the internal model with our custom model instance
    yolo.model = model
yolo.train(
    data=DATA_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    name="lf_yolo_weld_training",
    project="runs/detect",
    amp=AMP,
    plots=True,
    val=True,
    exist_ok=True,
    verbose=True
)

print("\nTraining complete!  Check 'runs/detect/lf_yolo_weld_training/' for results.")