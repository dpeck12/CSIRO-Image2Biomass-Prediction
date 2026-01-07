import os
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ---- Competition specifics ----
TARGET_NAMES = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]
TARGET_INDEX = {t: i for i, t in enumerate(TARGET_NAMES)}
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

# ImageNet normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---- Metrics ----
def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str]) -> float:
    weights = {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5,
    }
    w = np.array([weights[t] for t in target_names], dtype=np.float64)
    eps = 1e-8
    wsum = np.sum(w) + eps
    yw = np.sum(w * y_true) / wsum
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - yw) ** 2) + eps
    return 1.0 - ss_res / ss_tot


# ---- Utils ----
def get_data_root() -> str:
    kaggle_root = "/kaggle/input/csiro-biomass"
    if os.path.exists(kaggle_root):
        return kaggle_root
    return "data"


def get_work_root() -> str:
    kaggle_work = "/kaggle/working"
    return kaggle_work if os.path.exists(kaggle_work) else "."


def pivot_train_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index=[
            "image_path",
            "Sampling_Date",
            "State",
            "Species",
            "Pre_GSHH_NDVI",
            "Height_Ave_cm",
        ],
        columns="target_name",
        values="target",
        aggfunc="first",
    ).reset_index()
    for t in TARGET_NAMES:
        if t not in wide.columns:
            wide[t] = np.nan
    wide = wide[[
        "image_path",
        "Sampling_Date",
        "State",
        "Species",
        "Pre_GSHH_NDVI",
        "Height_Ave_cm",
    ] + TARGET_NAMES]
    return wide


# ---- Datasets ----
class TrainDataset(Dataset):
    def __init__(self, df_wide: pd.DataFrame, images_root: str, image_size: int = 224, augment: bool = True, log_targets: bool = True):
        self.df = df_wide
        self.images_root = images_root
        self.image_size = image_size
        t = self.df[TARGET_NAMES].values.astype(np.float32)
        self.log_targets = log_targets
        if log_targets:
            t = np.log1p(t)
        self.targets = t
        self.paths = self.df["image_path"].tolist()
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        p = os.path.join(self.images_root, self.paths[idx])
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Simple RandomResizedCrop + HorizontalFlip
        h, w = img.shape[:2]
        if self.augment:
            scale = np.random.uniform(0.8, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            top = np.random.randint(0, max(1, h - new_h + 1))
            left = np.random.randint(0, max(1, w - new_w + 1))
            img = img[top:top+new_h, left:left+new_w]
            if np.random.rand() < 0.5:
                img = np.ascontiguousarray(np.flip(img, axis=1))
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        img = torch.from_numpy(img).permute(2, 0, 1)
        y = torch.from_numpy(self.targets[idx])
        return img, y


class TestRowDataset(Dataset):
    def __init__(self, df_test: pd.DataFrame, images_root: str, image_size: int = 224):
        self.df = df_test
        self.images_root = images_root
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        p = os.path.join(self.images_root, row["image_path"])  # e.g., <root>/test/ID....jpg
        img = cv2.imread(p)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = (img - IMG_MEAN) / IMG_STD
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, row["target_name"], row["sample_id"]


# ---- Model ----
class SimpleImageRegressor(nn.Module):
    def __init__(self, out_dim: int = 5):
        super().__init__()
        from torchvision import models
        backbone = models.resnet34(weights=None)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


# ---- Runner ----
@dataclass
class Config:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = int(os.environ.get("EPOCHS", "5"))
    lr: float = 1e-3
    num_workers: int = 0


def train_and_save(cfg: Config, data_root: str, work_root: str) -> Optional[str]:
    train_csv = os.path.join(data_root, "train.csv")
    if not os.path.exists(train_csv):
        print(f"No train.csv at {train_csv}; skipping training.")
        return None

    df_train = pd.read_csv(train_csv)
    df_wide = pivot_train_long_to_wide(df_train)
    # Require complete targets for simple training
    mask = (~df_wide[TARGET_NAMES].isna()).all(axis=1)
    df_wide = df_wide[mask]
    if len(df_wide) == 0:
        print("No rows with all targets; skipping training.")
        return None

    ds = TrainDataset(df_wide, images_root=data_root, image_size=cfg.image_size, augment=True, log_targets=True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleImageRegressor(out_dim=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # Weighted MSE aligned with competition target weights
    w_vec = torch.tensor([TARGET_WEIGHTS[t] for t in TARGET_NAMES], dtype=torch.float32)

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        count = 0
        for imgs, ys in loader:
            imgs = imgs.to(device)
            ys = ys.to(device)
            preds = model(imgs)
            # Apply weighted MSE on log targets
            loss = torch.mean(w_vec.to(device) * (preds - ys) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)
        avg_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")

    ckpt_path = os.path.join(work_root, "model.pt")
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
    return ckpt_path


def infer_and_submit(cfg: Config, data_root: str, work_root: str, ckpt_path: Optional[str]):
    test_csv = os.path.join(data_root, "test.csv")
    df_test = pd.read_csv(test_csv)

    out_path = os.path.join(work_root, "submission.csv")
    os.makedirs(work_root, exist_ok=True)

    # Always build a lightweight tabular model from image features as a strong baseline
    # This tends to yield >= 0 R^2 compared to constant or naive CNN with few epochs.
    def extract_feats(img: np.ndarray) -> np.ndarray:
        # Assumes RGB float32 in [0,1]
        r = img[..., 0]; g = img[..., 1]; b = img[..., 2]
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        h = hsv[..., 0]; s = hsv[..., 1]; v = hsv[..., 2]
        eps = 1e-6
        gr = np.mean(g) / (np.mean(r) + eps)
        vi1 = (np.mean(g) - np.mean(r)) / (np.mean(g) + np.mean(r) + eps)
        vi2 = (2*np.mean(g) - np.mean(r) - np.mean(b)) / (2*np.mean(g) + np.mean(r) + np.mean(b) + eps)
        # Texture via Laplacian variance on grayscale
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        tex = float(lap.var())
        feats = [
            float(r.mean()), float(g.mean()), float(b.mean()),
            float(r.std()), float(g.std()), float(b.std()),
            float(h.mean()), float(s.mean()), float(v.mean()),
            float(h.std()), float(s.std()), float(v.std()),
            float(gr), float(vi1), float(vi2), float(tex)
        ]
        return np.asarray(feats, dtype=np.float32)

    # Build training features from df_wide
    train_csv = os.path.join(data_root, "train.csv")
    df_train = pd.read_csv(train_csv)
    df_wide = pivot_train_long_to_wide(df_train)
    mask = (~df_wide[TARGET_NAMES].isna()).all(axis=1)
    df_wide = df_wide[mask]
    X_feats = []
    Y = df_wide[TARGET_NAMES].values.astype(np.float32)
    # Compute per-target caps to keep predictions in plausible ranges
    caps_low = np.percentile(Y, 1, axis=0)
    caps_high = np.percentile(Y, 99, axis=0)
    meds = np.median(Y, axis=0)

    # Safe baseline option: predict per-target medians only
    if os.environ.get("SAFE_BASELINE", "0") == "1":
        out_path = os.path.join(work_root, "submission.csv")
        os.makedirs(work_root, exist_ok=True)
        preds = []
        for _, row in df_test.iterrows():
            idx = TARGET_INDEX[row["target_name"]]
            preds.append(float(meds[idx]))
        sub = df_test[["sample_id"]].copy()
        sub["target"] = preds
        sub.to_csv(out_path, index=False)
        print(f"Wrote SAFE_BASELINE submission to {out_path}")
        return
    for p in df_wide["image_path"].tolist():
        ip = os.path.join(data_root, p)
        im = cv2.imread(ip)
        if im is None:
            im = np.zeros((cfg.image_size, cfg.image_size, 3), dtype=np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (cfg.image_size, cfg.image_size)).astype(np.float32) / 255.0
        # For tabular features use raw [0,1] values (no ImageNet normalization)
        X_feats.append(extract_feats(im))
    X_feats = np.stack(X_feats, axis=0)
    # Log-transform targets
    Y_log = np.log1p(Y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_feats)
    # Train per-target ridge models
    ridges: list[Ridge] = []
    for i in range(len(TARGET_NAMES)):
        r = Ridge(alpha=1.0)
        r.fit(Xs, Y_log[:, i])
        ridges.append(r)

    # Predict for test using tabular baseline
    def predict_tabular_for_row(img: np.ndarray, tname: str) -> float:
        im = cv2.resize(img, (cfg.image_size, cfg.image_size)).astype(np.float32) / 255.0
        xf = extract_feats(im)[None, :]
        xf = scaler.transform(xf)
        idx = TARGET_INDEX[tname]
        ylog = ridges[idx].predict(xf)[0]
        pred = float(np.expm1(ylog))
        # Shrink toward per-target median for stability
        pred = 0.5 * pred + 0.5 * float(meds[idx])
        # Clamp to [0, caps]
        pred = max(0.0, pred)
        pred = float(np.clip(pred, caps_low[idx], caps_high[idx]))
        return pred

    use_cnn = ckpt_path is not None and os.path.exists(ckpt_path)
    if use_cnn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleImageRegressor(out_dim=5).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        model.eval()

    ds = TestRowDataset(df_test, images_root=data_root, image_size=cfg.image_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    preds = []
    with torch.no_grad():
        for img, tname, sid in loader:
            # CNN prediction (if available)
            pred_cnn = None
            if use_cnn:
                img_dev = img.to(device)
                out = model(img_dev).squeeze(0).cpu().numpy()
                idx = TARGET_INDEX[str(tname[0])]
                pred_cnn = float(np.expm1(out[idx]))
                # Shrink toward median
                pred_cnn = 0.5 * pred_cnn + 0.5 * float(meds[idx])
                pred_cnn = max(0.0, pred_cnn)
                pred_cnn = float(np.clip(pred_cnn, caps_low[idx], caps_high[idx]))
            # Tabular baseline prediction
            img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_tab = predict_tabular_for_row(img_np, str(tname[0]))
            # Blend if CNN exists
            if pred_cnn is not None:
                # Heavier weight on tabular baseline for stability
                pred = 0.2 * pred_cnn + 0.8 * pred_tab
            else:
                pred = pred_tab
            preds.append(pred)

    sub = df_test[["sample_id"]].copy()
    sub["target"] = preds
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    data_root = get_data_root()
    work_root = get_work_root() if os.path.exists("/kaggle/input") else "outputs"
    cfg = Config()
    ckpt = train_and_save(cfg, data_root, work_root)
    infer_and_submit(cfg, data_root, work_root, ckpt)
