import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from lf_yolo import LFYOLO_WeldDefect
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

# --- Config ---
DATA_CONFIG_PATH = "weld_defects/weld_defects.yaml"
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3

print(f"🚀 Training on {DEVICE.upper()} for {EPOCHS} epochs")

# --- Load YAML Config Properly ---
with open(DATA_CONFIG_PATH, "r") as f:
    data_cfg = yaml.safe_load(f)

if not all(k in data_cfg for k in ("train", "val", "nc", "names")):
    raise ValueError("Your YAML file must contain 'train', 'val', 'nc', and 'names' fields!")

print(f"📂 Found {data_cfg['nc']} classes: {data_cfg['names']}")

# --- Dataset ---
train_data = YOLODataset(
    data=data_cfg,  # ✅ pass dict instead of str
    task="detect",
    imgsz=IMG_SIZE,
    split="train",
    augment=True
)
val_data = YOLODataset(
    data=data_cfg,
    task="detect",
    imgsz=IMG_SIZE,
    split="val"
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# --- Model ---
model = LFYOLO_WeldDefect(nc=data_cfg["nc"]).to(DEVICE)
criterion = nn.MSELoss()  # placeholder loss
optimizer = optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

    for imgs, targets, paths, _ in pbar:
        imgs = imgs.to(DEVICE).float() / 255.0

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            preds = model(imgs)
            loss = criterion(preds[0], preds[0])  # dummy loss for now

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    print(f"✅ Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_loss:.4f}")

print("🎯 Training Complete! Model saved as 'lf_yolo_trained.pt'")
torch.save(model.state_dict(), "lf_yolo_trained.pt")
