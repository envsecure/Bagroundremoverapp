import os
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from tqdm import tqdm

from bg_remover.src.model.deeplab import deeplabv3_plus
from bg_remover.src.model.loss_fucn import dice_loss, dice_coef, iou
from bg_remover.src.data.data_for_train import train_data

# ── GPU Setup ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected — training will run on CPU.")
print(f"   Using device: {device}")
# ───────────────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DICT = ROOT_DIR / "configs" / "train.yaml"
MODEL_DICT = ROOT_DIR / "configs" / "model.yaml"

with open(TRAIN_DICT) as f:
    train_cfg = yaml.safe_load(f)
with open(MODEL_DICT) as f:
    model_cfg = yaml.safe_load(f)

# ── Model ──────────────────────────────────────────────────────────────────────
model = deeplabv3_plus((model_cfg["H"], model_cfg["W"], 3))
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

# ── Data ───────────────────────────────────────────────────────────────────────
train_loader, valid_loader = train_data()

num_epochs = train_cfg["num_epochs"]

# ── Training Loop ──────────────────────────────────────────────────────────────
for epoch in range(num_epochs):
    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    train_dice = 0.0
    train_iou = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # Forward
        preds = model(images)
        loss = dice_loss(preds, masks)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            d = dice_coef(preds, masks).item()
            i = iou(preds, masks).item()

        train_loss += loss.item()
        train_dice += d
        train_iou += i
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}", iou=f"{i:.4f}")

    n_train = len(train_loader)
    train_loss /= n_train
    train_dice /= n_train
    train_iou /= n_train

    # ── Validation ─────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]"):
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = dice_loss(preds, masks)

            val_loss += loss.item()
            val_dice += dice_coef(preds, masks).item()
            val_iou += iou(preds, masks).item()

    n_val = len(valid_loader)
    val_loss /= n_val
    val_dice /= n_val
    val_iou /= n_val

    print(
        f"\nEpoch {epoch+1}/{num_epochs} — "
        f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}  ||  "
        f"Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}\n"
    )

# ── Save Model ─────────────────────────────────────────────────────────────────
save_path = os.path.join(model_cfg["model_save_path"], "my_model.pth")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print(f"✅ Model saved to {save_path}")