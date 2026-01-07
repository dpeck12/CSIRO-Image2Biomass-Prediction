import torch

TARGET_NAMES = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]
TARGET_WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


def weighted_r2(y_true: torch.Tensor, y_pred: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted R^2 across a batch.
    Args:
        y_true: (N,) ground-truth values
        y_pred: (N,) predicted values
        weights: (N,) per-row weights
    Returns:
        scalar tensor
    """
    # To avoid division by zero, add small epsilon
    eps = 1e-8
    wsum = torch.sum(weights) + eps
    y_bar = torch.sum(weights * y_true) / wsum
    ss_res = torch.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = torch.sum(weights * (y_true - y_bar) ** 2) + eps
    r2 = 1.0 - ss_res / ss_tot
    return r2


def get_target_weights_tensor(target_names: list[str]) -> torch.Tensor:
    return torch.tensor([TARGET_WEIGHTS[t] for t in target_names], dtype=torch.float32)
