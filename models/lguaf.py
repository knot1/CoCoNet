import torch
import torch.nn as nn


class LGUAF(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, F_fuse: torch.Tensor, C: torch.Tensor, D: torch.Tensor):
        conf = self.head(torch.cat([F_fuse, C, D], dim=1))
        return conf * F_fuse, conf
