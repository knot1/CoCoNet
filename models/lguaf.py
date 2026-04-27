import torch
import torch.nn as nn


class LGUAF(nn.Module):
    def __init__(self, channels):
        super().__init__()

        hidden = max(channels // 4, 32)

        self.conf_head = nn.Sequential(
            nn.Conv2d(channels + 2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, conflict_map, disagreement_map):
        x = torch.cat([feat, conflict_map, disagreement_map], dim=1)
        conf = self.conf_head(x)

        out = feat * conf
        out = self.refine(out)

        return out, conf