import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarWaveletConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: [B, out_channels, H/2, W/2]
        """

        if x.shape[-1] % 2 != 0:
            x = x[:, :, :, :-1]
        if x.shape[-2] % 2 != 0:
            x = x[:, :, :-1, :]

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        LL = (x00 + x01 + x10 + x11) / 4.0
        LH = (x00 - x01 + x10 - x11) / 4.0
        HL = (x00 + x01 - x10 - x11) / 4.0
        HH = (x00 - x01 - x10 + x11) / 4.0

        x_wave = torch.cat([LL, LH, HL, HH], dim=1)

        return self.proj(x_wave)


class ConflictSemanticModulation(nn.Module):
    def __init__(self, channels, num_classes, hidden=64):
        super().__init__()

        self.weight_gen = nn.Sequential(
            nn.Conv2d(1 + num_classes, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_s, F_f, S):
        """
        F_s: [B, C, H, W]  spatial feature
        F_f: [B, C, H, W]  frequency feature
        S:   [B, K]        CLIP semantic prior

        return:
            F_fuse: [B, C, H, W]
            C_map:  [B, 1, H, W]
            W:      [B, 1, H, W]
        """

        B, C, H, W = F_s.shape

        if F_f.shape[-2:] != (H, W):
            F_f = F.interpolate(
                F_f,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        if F_f.shape[1] != C:
            raise ValueError(f"Channel mismatch: F_s={F_s.shape}, F_f={F_f.shape}")

        f_s = F.normalize(F_s, dim=1)
        f_f = F.normalize(F_f, dim=1)

        C_map = 1.0 - torch.sum(f_s * f_f, dim=1, keepdim=True)

        S_map = S.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        W = self.weight_gen(torch.cat([C_map, S_map], dim=1))

        F_fuse = W * F_s + (1.0 - W) * F_f
        F_fuse = self.refine(F_fuse)

        return F_fuse, C_map, W