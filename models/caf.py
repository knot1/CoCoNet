import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_conflict_map(x_rgb: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
    conflict_map = 1.0 - torch.sum(
        F.normalize(x_rgb, dim=1) * F.normalize(x_e, dim=1),
        dim=1,
        keepdim=True
    )
    return torch.clamp(conflict_map, min=0.0, max=2.0)


class SimpleFusion(nn.Module):
    """
    Baseline fusion: concat(rgb, extra) -> 1x1 conv -> fused.
    """
    def __init__(self, c: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_rgb, x_e], dim=1)
        return self.proj(x)


class ConflictAwareFusionBlock(nn.Module):
    """
    Conflict-aware Adaptive Fusion (CAF).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.reweight = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        conflict_map = compute_conflict_map(x_rgb, x_e)
        conflict_norm = conflict_map / 2.0

        reweight = self.reweight(torch.cat([x_rgb, x_e], dim=1))
        rgb_weight = (1.0 - conflict_norm) * reweight
        e_weight = (1.0 - conflict_norm) * (1.0 - reweight)

        fused = self.fuse(torch.cat([x_rgb * rgb_weight, x_e * e_weight], dim=1))
        return fused, conflict_map
