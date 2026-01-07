import os
import yaml
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from biomass.data import BiomassTrainDataset
from biomass.model import ImageTabularRegressor
from biomass.utils import pivot_train_long_to_wide, TARGET_NAMES
from biomass.metrics import weighted_r2, TARGET_WEIGHTS


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    tabs = [b[1] for b in batch]
    if tabs and tabs[0].numel() > 0:
        tabs = torch.stack(tabs)
    else:
        tabs = None
    ys = torch.stack([b[2] for b in batch])
    return images, tabs, ys


def train(cfg_path: str = "configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    train_csv = os.path.join(data_root, cfg["train_csv"])
    images_root = data_root
    image_size = cfg.get("image_size", 224)
    batch_size = cfg.get("batch_size", 16)
    epochs = cfg.get("epochs", 5)
    lr = cfg.get("lr", 1e-3)
    seed = cfg.get("seed", 42)
    use_tabular = cfg.get("use_tabular", True)
    backbone = cfg.get("backbone", "efficientnet_b0")
    out_dir = cfg.get("out_dir", "outputs")
    num_workers = cfg.get("num_workers", 0)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(seed)

    df_train = pd.read_csv(train_csv)
    df_wide = pivot_train_long_to_wide(df_train)
    # Drop rows missing any target (rare)
    df_wide = df_wide.dropna(subset=TARGET_NAMES)

    train_df, val_df = train_test_split(df_wide, test_size=0.2, random_state=seed)

    train_ds = BiomassTrainDataset(train_df, images_root=images_root, image_size=image_size, use_tabular=use_tabular)
    val_ds = BiomassTrainDataset(val_df, images_root=images_root, image_size=image_size, use_tabular=use_tabular)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    device = get_device()
    model = ImageTabularRegressor(backbone_name=backbone, image_size=image_size, tabular_dim=(train_ds.tabular.shape[1] if use_tabular else 0), out_dim=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_r2 = -1.0
    ckpt_path = os.path.join(out_dir, "model.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, tabs, ys in train_loader:
            images = images.to(device)
            if tabs is not None:
                tabs = tabs.to(device)
            ys = ys.to(device)
            preds = model(images, tabs)
            loss = criterion(preds, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_true = []
        all_pred = []
        all_w = []
        with torch.no_grad():
            for images, tabs, ys in val_loader:
                images = images.to(device)
                if tabs is not None:
                    tabs = tabs.to(device)
                ys = ys.to(device)
                preds = model(images, tabs)
                loss = criterion(preds, ys)
                val_loss += loss.item() * images.size(0)
                # Flatten to long
                all_true.append(ys.view(-1))
                all_pred.append(preds.view(-1))
                # Build weights per target
                # Each row produces 5 targets in fixed order
                w = torch.tensor([TARGET_WEIGHTS[t] for t in TARGET_NAMES], dtype=torch.float32, device=device)
                w = w.unsqueeze(0).repeat(images.size(0), 1).view(-1)
                all_w.append(w)
        val_loss /= len(val_loader.dataset)
        y_true = torch.cat(all_true)
        y_pred = torch.cat(all_pred)
        w = torch.cat(all_w)
        r2 = weighted_r2(y_true, y_pred, w).item()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} weighted_r2={r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            torch.save({
                "model_state": model.state_dict(),
                "backbone": backbone,
                "image_size": image_size,
                "use_tabular": use_tabular,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    print(f"Best weighted R2: {best_r2:.4f}")


if __name__ == "__main__":
    train()
