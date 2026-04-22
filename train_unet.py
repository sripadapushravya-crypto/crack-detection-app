import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A

# =========================
# CONFIG
# =========================
PATCH_SIZES = [256, 320, 384]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATASET
# =========================
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train = train
        self.files = []

        for f in os.listdir(img_dir):
            if not f.endswith(".jpg"):
                continue

            base = os.path.splitext(f)[0]
            mask1 = os.path.join(mask_dir, base + ".png")
            mask2 = os.path.join(mask_dir, base + "_mask.png")

            if os.path.exists(mask1) or os.path.exists(mask2):
                self.files.append(f)

        print(f"Loaded {len(self.files)} samples from {img_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]

        img_path = os.path.join(self.img_dir, img_name)
        base = os.path.splitext(img_name)[0]

        mask1 = os.path.join(self.mask_dir, base + ".png")
        mask2 = os.path.join(self.mask_dir, base + "_mask.png")
        mask_path = mask1 if os.path.exists(mask1) else mask2

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if image is None or mask is None:
            raise ValueError(f"Corrupted file: {img_name}")

        # ===== MULTI-SCALE PATCH =====
        p = np.random.choice(PATCH_SIZES) if self.train else 320
        h, w = image.shape[:2]

        if self.train:
            if h < p or w < p:
                image = cv2.resize(image, (p, p))
                mask = cv2.resize(mask, (p, p))
            else:
                # stronger crack bias
                if np.random.rand() < 0.3:
                    # background
                    x = np.random.randint(0, w - p + 1)
                    y = np.random.randint(0, h - p + 1)
                else:
                    # crack-focused
                    for _ in range(15):
                        x = np.random.randint(0, w - p + 1)
                        y = np.random.randint(0, h - p + 1)
                        if mask[y:y+p, x:x+p].sum() > 200:
                            break

                image = image[y:y+p, x:x+p]
                mask = mask[y:y+p, x:x+p]

        # ===== AUGMENT =====
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.2),
            A.Resize(320, 320)
        ]) if self.train else A.Compose([
            A.Resize(320, 320)
        ])

        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        # ===== NORMALIZE =====
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)


# =========================
# METRIC
# =========================
def iou_score(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection

    return (intersection + 1e-6) / (union + 1e-6)


# =========================
# MAIN TRAIN
# =========================
def main():

    train_dataset = CrackDataset("data/train/images", "data/train/masks", True)
    val_dataset = CrackDataset("data/val/images", "data/val/masks", False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2)

    # ===== MODEL =====
    model = smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(DEVICE)

    # ===== FREEZE SHORT =====
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ===== LOSSES =====
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.0]).to(DEVICE))
    dice = smp.losses.DiceLoss(mode="binary")
    focal = smp.losses.FocalLoss(mode="binary")

    def loss_fn(preds, masks):
        return (
            0.4 * bce(preds, masks)
            + 0.4 * dice(preds, masks)
            + 0.2 * focal(preds, masks)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=60
    )

    scaler = torch.amp.GradScaler("cuda")

    best_iou = 0
    early_stop = 0

    for epoch in range(60):

        # ===== UNFREEZE EARLY =====
        if epoch == 2:
            for param in model.encoder.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g['lr'] = 5e-5

        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                preds = model(images)
                loss = loss_fn(preds, masks)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                preds = model(images)
                val_iou += iou_score(preds, masks).item()

        val_iou /= len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.2f} | IoU: {val_iou:.4f}")

        scheduler.step()

        # ===== SAVE BEST =====
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch
            }, "best_model.pth")
            print("🔥 Saved best model")
            early_stop = 0
        else:
            early_stop += 1

        # ===== LONGER PATIENCE =====
        if early_stop >= 15:
            print("⛔ Early stopping")
            break


if __name__ == "__main__":
    main()