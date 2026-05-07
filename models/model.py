import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

import torch
import torch.nn as nn
import torch_dct as DCT

import matplotlib.pyplot as plt
import numpy as np
import torch_dct as DCT
import os

import torch
import torch_dct as DCT
import matplotlib.pyplot as plt
import os

from losses.dense_proto_loss import dense_proto_alignment_loss
from .modules import ModalitySemanticProjector
from .prompt_semantic import _build_prompts

def denormalize_dsm(x):
    mean = torch.tensor([0.5]).view(1,1,1,1).to(x.device)
    std = torch.tensor([0.5]).view(1,1,1,1).to(x.device)
    return x * std + mean

def denormalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
    return x * std + mean




def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


class Baseline(nn.Module):
    def __init__(
        self,
        cfg=None,
        num_classes=None,
        norm_layer=nn.BatchNorm2d,
        in_chans=None,
        class_labels=None,
        vlsp_cfg=None,
    ):
        super(Baseline, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        if in_chans is not None:
            self.in_chans = in_chans
        else:
            self.in_chans = [3, 1]

        if cfg.backbone == 'mit_b5':
            from .encoder import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b4':
            from .encoder import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b2':
            from .encoder import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b1':
            from .encoder import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b0':
            from .encoder import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
            self.channels = [32, 64, 160, 256]
        else:
            from .encoder import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)

        from .Seg_head import DecoderHead
        self.decode_head = DecoderHead(in_channels=self.channels, num_classes=num_classes, norm_layer=norm_layer,
                                       embed_dim=cfg.decoder_embed_dim)

        self.prompt_semantic = None
        prompt_cfg = getattr(cfg, "prompt_semantic", None)
        if prompt_cfg is not None and getattr(prompt_cfg, "enabled", False):
            if class_labels is None:
                class_labels = [str(idx) for idx in range(num_classes)]
            from .prompt_semantic import PromptSemanticPrior
            self.prompt_semantic = PromptSemanticPrior(
                class_labels=class_labels,
                model_name=prompt_cfg.model_name,
                pretrained=prompt_cfg.pretrained,
                image_size=prompt_cfg.image_size,
            )

        self.vlsp_cfg = vlsp_cfg
        self.vlsp_enabled = False
        self.vlsp_use_modality_projection = True
        self.vlsp_use_dense = False
        self.vlsp_use_shared = False
        self.vlsp_use_adaptive = False
        self.vlsp_use_regularization = False
        self.vlsp_align_stage = "stage4"
        self.vlsp_ignore_classes = []
        self.vlsp_ignore_index = None
        self.clip_embed_dim = None

        if vlsp_cfg is not None and getattr(vlsp_cfg, "ENABLE", False):
            if class_labels is None:
                class_labels = [str(idx) for idx in range(num_classes)]
            self.vlsp_enabled = True
            self.vlsp_use_modality_projection = getattr(vlsp_cfg, "USE_MODALITY_PROJECTION", True)
            self.vlsp_use_dense = getattr(vlsp_cfg, "USE_DENSE_ALIGNMENT", True)
            self.vlsp_use_shared = getattr(vlsp_cfg, "USE_SHARED_PROTOTYPE", False)
            self.vlsp_use_adaptive = getattr(vlsp_cfg, "USE_ADAPTIVE_PROTOTYPE", False)
            self.vlsp_use_regularization = getattr(vlsp_cfg, "USE_REGULARIZATION", False)
            self.vlsp_align_stage = getattr(vlsp_cfg, "ALIGN_STAGE", "stage4")

            ignore_background = getattr(vlsp_cfg, "IGNORE_BACKGROUND_ALIGNMENT", False)
            self.vlsp_ignore_classes = self._resolve_ignore_classes(class_labels, ignore_background)

            clip_model_name = "ViT-B-32"
            clip_pretrained = "openai"
            if prompt_cfg is not None:
                clip_model_name = getattr(prompt_cfg, "model_name", clip_model_name)
                clip_pretrained = getattr(prompt_cfg, "pretrained", clip_pretrained)

            clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name=clip_model_name,
                pretrained=clip_pretrained,
            )
            clip_model.eval()
            for param in clip_model.parameters():
                param.requires_grad = False
            self.clip_model = clip_model
            self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

            base_prototypes = self._build_vlsp_prototypes(class_labels)
            self.register_buffer("base_prototypes", base_prototypes)
            self.clip_embed_dim = base_prototypes.size(1)

            if self.vlsp_use_adaptive:
                self.prototype_delta = nn.Parameter(torch.zeros_like(base_prototypes))
            else:
                self.register_buffer("prototype_delta", torch.zeros_like(base_prototypes), persistent=False)

            self.modality_projector = ModalitySemanticProjector(self.clip_embed_dim)
            stage_channels = self.channels[self._resolve_align_stage(self.vlsp_align_stage)]
            self.rgb_semantic_head = nn.Conv2d(stage_channels, self.clip_embed_dim, 1)
            self.dsm_semantic_head = nn.Conv2d(stage_channels, self.clip_embed_dim, 1)

        self.init_weights(cfg, pretrained=cfg.pretrained_backbone)


    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

    @staticmethod
    def _resolve_align_stage(stage_name: str) -> int:
        stage_map = {
            "stage1": 0,
            "stage2": 1,
            "stage3": 2,
            "stage4": 3,
        }
        if stage_name not in stage_map:
            raise ValueError(f"Unknown ALIGN_STAGE '{stage_name}'")
        return stage_map[stage_name]

    @staticmethod
    def _resolve_ignore_classes(class_labels, ignore_background: bool):
        if not ignore_background or not class_labels:
            return []
        ignore_tokens = {"background", "clutter", "others", "other", "undefined", "unlabeled"}
        return [
            idx
            for idx, label in enumerate(class_labels)
            if label.strip().lower() in ignore_tokens
        ]

    def _build_vlsp_prototypes(self, class_labels):
        device = next(self.clip_model.parameters()).device
        prototypes = []
        with torch.no_grad():
            for label in class_labels:
                prompts = _build_prompts(label)
                tokens = self.clip_tokenizer(prompts).to(device)
                text_features = self.clip_model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=-1)
                proto = text_features.mean(dim=0)
                proto = F.normalize(proto, dim=-1)
                prototypes.append(proto.detach().cpu())
        return torch.stack(prototypes, dim=0)

    def _get_vlsp_prototypes(self):
        prototypes = self.base_prototypes
        if self.vlsp_use_adaptive:
            prototypes = prototypes + self.prototype_delta
        prototypes = F.normalize(prototypes, dim=-1)
        if self.vlsp_use_shared or not self.vlsp_use_modality_projection:
            return prototypes, prototypes
        return self.modality_projector(prototypes)

    def _project_modal_features(self, rgb_feat, dsm_feat):
        if self.vlsp_use_modality_projection or rgb_feat.size(1) != self.clip_embed_dim:
            rgb_embed = self.rgb_semantic_head(rgb_feat)
        else:
            rgb_embed = rgb_feat
        if self.vlsp_use_modality_projection or dsm_feat.size(1) != self.clip_embed_dim:
            dsm_embed = self.dsm_semantic_head(dsm_feat)
        else:
            dsm_embed = dsm_feat
        rgb_embed = F.normalize(rgb_embed, dim=1)
        dsm_embed = F.normalize(dsm_embed, dim=1)
        return rgb_embed, dsm_embed

    def encode_decode(self, rgb, modal_x, return_modal_features: bool = False):
        ori_size = rgb.shape
        if return_modal_features:
            x_semantic, L_cons, low_L_cons, modal_features = self.backbone(
                rgb, modal_x, return_modal_features=True, align_stage=self.vlsp_align_stage
            )
        else:
            x_semantic, L_cons, low_L_cons = self.backbone(rgb, modal_x)
            modal_features = None

        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)

        return out_semantic, L_cons, low_L_cons, modal_features

    def forward(self, rgb, modal_x, mask=None):
        if modal_x.ndim == 3:
            modal_x = torch.unsqueeze(modal_x, dim=1)
        return_modal = self.vlsp_enabled and self.vlsp_use_dense
        outputs, L_cons, low_L_cons, modal_features = self.encode_decode(
            rgb, modal_x, return_modal_features=return_modal
        )
        semantic_prior = self.prompt_semantic(rgb) if self.prompt_semantic is not None else None

        loss_rgb = outputs.new_zeros(())
        loss_dsm = outputs.new_zeros(())
        loss_reg = outputs.new_zeros(())

        if self.vlsp_enabled:
            if self.vlsp_use_regularization and self.vlsp_use_adaptive:
                loss_reg = (self.prototype_delta ** 2).sum()

            if self.vlsp_use_dense and modal_features is not None and mask is not None:
                rgb_feat, dsm_feat = modal_features
                rgb_embed, dsm_embed = self._project_modal_features(rgb_feat, dsm_feat)
                proto_rgb, proto_dsm = self._get_vlsp_prototypes()

                mask = mask.long()
                if mask.shape[-2:] != rgb_embed.shape[-2:]:
                    mask = F.interpolate(mask.unsqueeze(1).float(),
                                         size=rgb_embed.shape[-2:],
                                         mode="nearest").squeeze(1).long()

                loss_rgb = dense_proto_alignment_loss(
                    rgb_embed, proto_rgb, mask,
                    ignore_index=self.vlsp_ignore_index,
                    ignore_classes=self.vlsp_ignore_classes,
                )
                loss_dsm = dense_proto_alignment_loss(
                    dsm_embed, proto_dsm, mask,
                    ignore_index=self.vlsp_ignore_index,
                    ignore_classes=self.vlsp_ignore_classes,
                )

        vlsp_losses = {
            "loss_rgb": loss_rgb,
            "loss_dsm": loss_dsm,
            "loss_reg": loss_reg,
        }

        return outputs, L_cons, low_L_cons, semantic_prior, vlsp_losses
