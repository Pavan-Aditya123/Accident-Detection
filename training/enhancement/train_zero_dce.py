"""
Zero-DCE (Zero-Reference Deep Curve Estimation) Training Script.

Training Pipeline & Method Overview:
- Input: Low-light images from datasets/low_light/our485/low.
- Zero-Reference Learning: Unlike supervised image-to-image translation methods (e.g. RetinexNet or Pix2Pix),
  Zero-DCE does NOT require paired ground-truth high-light images. It formulates low-light enhancement as a task
  of estimating image-specific higher-order light curves using set non-reference physical loss functions:
    1. Spatial Consistency Loss (L_spa) - preserves local spatial gradient structures.
    2. Exposure Control Loss (L_exp) - adjusts patch-level exposure toward optimal target (E = 0.6).
    3. Color Constancy Loss (L_col) - balances inter-channel RGB ratios (Gray-World assumption).
    4. Illumination Smoothness Loss (L_tv) - maintains monotonic and smooth curve parameter maps.

Why High-Light Images Are Not Used:
Paired datasets (low-light vs. normal-light) are difficult to capture in real-world CCTV environments due to motion, 
dynamic lighting changes, and ghosting artifacts. Zero-DCE eliminates paired data dependence entirely, enabling superior
generalizability to unseen night-time traffic feeds.

Pipeline Integration with YOLO26:
    Low-Light CCTV Frame
             ↓
    Zero-DCE Dynamic Enhancement (best_zero_dce.pth)
             ↓
    YOLO26 Baseline Accident Detection (best.pt)
             ↓
    Accident Alert / Severity Response
"""

import os
from pathlib import Path
import sys
import time
from glob import glob
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Import DCE-Net architecture and Zero-DCE Loss module
from models.enhancement.zero_dce.model import DCENet, ZeroDCELoss


class ZeroDCEDataset(Dataset):
    """
    PyTorch Dataset for Zero-DCE Unsupervised Low-Light Image Enhancement.
    Loads low-light images exclusively without requiring paired target images.
    """
    def __init__(self, image_dir: str, size: int = 256):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")) + 
                                  glob(os.path.join(image_dir, "*.jpg")) + 
                                  glob(os.path.join(image_dir, "*.jpeg")))
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor


def main():
    # Resolve project root directory
    base_dir = Path(__file__).resolve().parent.parent.parent
    dataset_dir = base_dir / "datasets" / "low_light" / "our485" / "low"
    save_dir = base_dir / "models" / "enhancement" / "zero_dce"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  ZERO-DCE ZERO-REFERENCE LOW-LIGHT ENHANCEMENT MODEL TRAINING")
    print("=" * 65)
    print(f"[+] Dataset Path      : {dataset_dir}")
    print(f"[+] Save Directory    : {save_dir}")

    if not dataset_dir.exists():
        print(f"[!] Warning: Low-light dataset directory not found at '{dataset_dir}'.")
        print("    Please ensure datasets/low_light/our485/low exists with images.")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Compute Device    : {device}")

    # Dataset Setup
    dataset = ZeroDCEDataset(str(dataset_dir), size=256)
    if len(dataset) == 0:
        print("[!] Error: Dataset is empty. Cannot start training.")
        return

    # Reproducible 80/20 train/validation split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    print(f"[+] Total Images      : {len(dataset)} (Train: {train_size}, Val: {val_size})")

    # Initialize DCE-Net Model, Zero-DCE Loss, Optimizer & LR Scheduler
    model = DCENet().to(device)
    criterion = ZeroDCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
    
    # ReduceLROnPlateau scheduler for dynamic learning rate adjustment
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    epochs = 100
    best_val_loss = float("inf")
    
    print("\n[+] Starting Zero-DCE Training Loop (100 Epochs)...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for img in train_loader:
            img = img.to(device)
            optimizer.zero_grad()
            
            enhanced_img, A = model(img)
            loss, _ = criterion(img, enhanced_img, A)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img in val_loader:
                img = img.to(device)
                enhanced_img, A = model(img)
                loss, _ = criterion(img, enhanced_img, A)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        
        # Step LR Scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.2f}s")

        # Save last checkpoint
        last_ckpt_path = save_dir / "last_zero_dce.pth"
        torch.save(model.state_dict(), last_ckpt_path)

        # Save best checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = save_dir / "best_zero_dce.pth"
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"    --> Saved new best model checkpoint: '{best_ckpt_path.name}' (Val Loss: {val_loss:.4f})")

    print("\n" + "=" * 65)
    print("      ZERO-DCE TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"[+] Best Model Saved : {save_dir / 'best_zero_dce.pth'}")
    print(f"[+] Last Model Saved : {save_dir / 'last_zero_dce.pth'}")

if __name__ == "__main__":
    main()
