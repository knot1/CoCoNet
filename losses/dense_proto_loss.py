import torch
import torch.nn.functional as F


def dense_proto_alignment_loss(feat, proto, mask, ignore_index=None, ignore_classes=None):
    if feat.dim() != 4:
        raise ValueError(f"Expected feat with 4 dims [B, D, H, W], got {feat.shape}")
    if proto.dim() != 2:
        raise ValueError(f"Expected proto with 2 dims [K, D], got {proto.shape}")
    if mask.dim() != 3:
        raise ValueError(f"Expected mask with 3 dims [B, H, W], got {mask.shape}")

    b, d, h, w = feat.shape
    k, proto_dim = proto.shape
    if d != proto_dim:
        raise ValueError(f"Feature dim {d} does not match prototype dim {proto_dim}")

    ignore_set = set(ignore_classes or [])
    if ignore_index is not None:
        ignore_set.add(ignore_index)

    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, d)
    mask_flat = mask.reshape(-1)

    total_loss = feat.new_zeros(())
    valid_classes = 0
    for class_idx in range(k):
        if class_idx in ignore_set:
            continue
        class_mask = mask_flat == class_idx
        if not torch.any(class_mask):
            continue
        feat_c = feat_flat[class_mask]
        proto_c = proto[class_idx].unsqueeze(0)
        sim = F.cosine_similarity(feat_c, proto_c, dim=-1, eps=1e-6)
        total_loss = total_loss + (1.0 - sim).mean()
        valid_classes += 1

    if valid_classes == 0:
        return total_loss
    return total_loss / valid_classes
