import os
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader

from biomass.data import BiomassTestDataset
from biomass.model import ImageTabularRegressor
from biomass.metrics import TARGET_NAMES


def collate_fn_test(batch):
    images = torch.stack([b[0] for b in batch])
    target_names = [b[2] for b in batch]
    return images, target_names


def infer(cfg_path: str = "configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    data_root = cfg["data_root"]
    test_csv = os.path.join(data_root, cfg["test_csv"])
    images_root = data_root
    image_size = cfg.get("image_size", 224)
    out_dir = cfg.get("out_dir", "outputs")
    ckpt_path = os.path.join(out_dir, "model.pt")

    df_test = pd.read_csv(test_csv)
    ds = BiomassTestDataset(df_test, images_root=images_root, image_size=image_size, use_tabular=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn_test)

    if not os.path.exists(ckpt_path):
        # Fallback: produce a baseline submission of zeros if no checkpoint available
        sub = df_test[["sample_id"]].copy()
        sub["target"] = 0.0
        os.makedirs(out_dir, exist_ok=True)
        sub_path = os.path.join(out_dir, "submission.csv")
        sub.to_csv(sub_path, index=False)
        print(f"No checkpoint found at {ckpt_path}. Wrote zero-baseline submission to {sub_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ImageTabularRegressor(backbone_name=ckpt["backbone"], image_size=ckpt["image_size"], tabular_dim=(5 if ckpt["use_tabular"] else 0), out_dim=5)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds_rows = []
    with torch.no_grad():
        for images, target_names in loader:
            outputs = model(images, None)  # no tabular at test
            outputs = outputs.cpu().numpy()
            # For each image in batch, replicate 5 targets
            for i in range(images.size(0)):
                image_targets = outputs[i]
                # df_test has rows per (image, target_name)
                # We need to pick the predicted component matching target_name
                # We'll map target_name to index in TARGET_NAMES
                # The DataLoader preserves order of df_test
                # For the current batch item, target_names[i] is the current requested component
                # But loader batches multiple rows per image; ensure we emit per row
                # Simpler: iterate paired over batch rows (one row per image-target in our dataset)
                pass
    # The above approach is complicated because our dataset built per row isn't aligned.
    # Alternative: perform per-row inference with batch size 1 for simplicity and correctness.

    # Rebuild a row-wise loader
    ds_rowwise = BiomassTestDataset(df_test, images_root=images_root, image_size=image_size, use_tabular=False)
    loader_row = DataLoader(ds_rowwise, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn_test)

    preds_rows = []
    with torch.no_grad():
        for images, target_names in loader_row:
            outputs = model(images, None)
            outputs = outputs.squeeze(0).cpu().numpy()  # 5 outputs
            tname = target_names[0]
            tindex = TARGET_NAMES.index(tname)
            pred = float(outputs[tindex])
            preds_rows.append(pred)

    # Build submission
    sub = df_test[["sample_id"]].copy()
    sub["target"] = preds_rows
    os.makedirs(out_dir, exist_ok=True)
    sub_path = os.path.join(out_dir, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Wrote submission to {sub_path}")


if __name__ == "__main__":
    infer()
