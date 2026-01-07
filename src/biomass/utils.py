import os
from typing import Tuple

import numpy as np
import pandas as pd

TARGET_NAMES = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_train_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def pivot_train_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long-format train.csv (multiple rows per image with target_name/target)
    into wide format: one row per image with one column per target.
    """
    # Important: do NOT include sample_id in the index here, since sample_id encodes the target_name
    # and will prevent grouping the five target rows into a single image-level row.
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
    # Ensure consistent column order
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


def parse_date_to_ordinal(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").map(lambda d: d.toordinal() if pd.notnull(d) else np.nan)

