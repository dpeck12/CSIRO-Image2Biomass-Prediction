import torch
import torch.nn as nn
import timm


class ImageTabularRegressor(nn.Module):
    def __init__(self, backbone_name: str = "efficientnet_b0", image_size: int = 224, tabular_dim: int = 5, out_dim: int = 5):
        super().__init__()
        # Use pretrained=False to avoid internet downloads in local/offline environments
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        in_features = self.backbone.num_features
        self.use_tabular = tabular_dim > 0
        fusion_dim = in_features + (tabular_dim if self.use_tabular else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.backbone(image)
        if self.use_tabular and tabular is not None and tabular.numel() > 0:
            x = torch.cat([feats, tabular], dim=1)
        else:
            x = feats
        out = self.head(x)
        return out
