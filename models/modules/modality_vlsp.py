import torch.nn as nn
import torch.nn.functional as F


class ModalitySemanticProjector(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.rgb_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dsm_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, base_prototypes):
        p_rgb = F.normalize(self.rgb_proj(base_prototypes), dim=-1)
        p_dsm = F.normalize(self.dsm_proj(base_prototypes), dim=-1)
        return p_rgb, p_dsm
