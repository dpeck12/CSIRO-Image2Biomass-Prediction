from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import TARGET_NAMES, parse_date_to_ordinal, pivot_train_long_to_wide


@dataclass
class DataConfig:
    images_root: str
    train_csv: str
    image_size: int = 224
    use_tabular: bool = True


class BiomassTrainDataset(Dataset):
    def __init__(self, df_wide: pd.DataFrame, images_root: str, image_size: int = 224, use_tabular: bool = True, aug: Optional[object] = None):
        self.df = df_wide
        self.images_root = images_root
        self.image_size = image_size
        self.use_tabular = use_tabular
        self.aug = aug

        # Build tabular features
        date_ord = parse_date_to_ordinal(self.df["Sampling_Date"]).astype(float).fillna(self.df["Sampling_Date"].notnull().sum())
        ndvi = self.df["Pre_GSHH_NDVI"].astype(float).fillna(self.df["Pre_GSHH_NDVI"].median())
        height = self.df["Height_Ave_cm"].astype(float).fillna(self.df["Height_Ave_cm"].median())
        # Simple categorical encodings
        state_cat = self.df["State"].astype("category").cat.codes.astype(float)
        species_cat = self.df["Species"].astype("category").cat.codes.astype(float)
        self.tabular = np.stack([date_ord, ndvi, height, state_cat, species_cat], axis=1).astype(np.float32)
        # Targets
        self.targets = self.df[TARGET_NAMES].values.astype(np.float32)
        # Image paths
        self.image_paths = self.df["image_path"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = f"{self.images_root}/{self.image_paths[idx]}"
        img = cv2.imread(img_path)
        if img is None:
            # If path is like train/ID.jpg, keep as is
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        if self.aug is not None:
            augmented = self.aug(image=img)
            img = augmented["image"]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.use_tabular:
            tab = torch.from_numpy(self.tabular[idx])
        else:
            tab = torch.empty(0)

        y = torch.from_numpy(self.targets[idx])
        return img, tab, y


class BiomassTestDataset(Dataset):
    def __init__(self, df_test: pd.DataFrame, images_root: str, image_size: int = 224, use_tabular: bool = True, aug: Optional[object] = None):
        self.df = df_test
        self.images_root = images_root
        self.image_size = image_size
        self.use_tabular = use_tabular
        self.aug = aug
        # Build tabular features (subset available in test.csv)
        # test.csv likely doesn't include tabular features beyond image_path and target_name
        # For baseline, we skip tabular for test
        self.image_paths = self.df["image_path"].tolist()
        self.target_names = self.df["target_name"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = f"{self.images_root}/{self.image_paths[idx]}"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        if self.aug is not None:
            augmented = self.aug(image=img)
            img = augmented["image"]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        # No tabular features for baseline
        tab = torch.empty(0)
        return img, tab, self.target_names[idx]
