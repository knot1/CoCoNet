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


# class ConflictAwareFusionBlock(nn.Module):
#     """
#     Conflict-aware Adaptive Fusion (CAF).
#     """
#     def __init__(self, channels: int):
#         super().__init__()
#         self.reweight = nn.Sequential(
#             nn.Conv2d(channels * 2, 1, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
#         self.fuse = nn.Sequential(
#             nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
#         conflict_map = compute_conflict_map(x_rgb, x_e)
#         conflict_norm = conflict_map / 2.0

#         reweight = self.reweight(torch.cat([x_rgb, x_e], dim=1))
#         rgb_weight = (1.0 - conflict_norm) * reweight
#         e_weight = (1.0 - conflict_norm) * (1.0 - reweight)

#         fused = self.fuse(torch.cat([x_rgb * rgb_weight, x_e * e_weight], dim=1))
#         return fused, conflict_map
class ConflictAwareFusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # 把 conflict_map (1通道) 也作为输入 concat 进去，让网络参考冲突程度来分配权重
        self.reweight = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, kernel_size=1, bias=False), # 输出2个通道，分别给RGB和Extra
            nn.Softmax(dim=1) # 保证加起来等于1，绝对不会出现双双归零的情况
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        conflict_map = compute_conflict_map(x_rgb, x_e) # [B, 1, H, W]
        
        # 将冲突图和两个特征拼接，计算自适应权重
        weight_input = torch.cat([x_rgb, x_e, conflict_map], dim=1)
        weights = self.reweight(weight_input) # [B, 2, H, W]
        rgb_weight = weights[:, 0:1, :, :]
        e_weight = weights[:, 1:2, :, :]

        # 加权融合
        fused = self.fuse(torch.cat([x_rgb * rgb_weight, x_e * e_weight], dim=1))
        return fused, conflict_map
