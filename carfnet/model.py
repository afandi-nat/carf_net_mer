"""CARF-Net: Class-Aware Region Fusion Network.

A shared ResNet-18 stem feeds two region branches (upper face / lower face).
The Area Weighting Module (AWM) re-weights the concatenated branch features,
three linear heads produce logits, and a class-aware gate fuses the branch
logits in logit space.

    dual-branch ResNet-18   -> base_layers + eyes_branch / mouth_branch
    Area Weighting Module   -> area_weight, in feature space
    class-aware fusion      -> fusion_gate (one gate per class) + joint_gate,
                               in logit space

Design notes, with roughly 150-700 training clips per dataset:

* Joint head. Summing two branch logits cannot represent AU combinations that
  span regions (disgust pairs AU4 around the eyes with AU9/AU10 around the
  mouth). The joint head sees the concatenated features and costs only
  1024 x num_classes parameters.
* Learned logit scale. Branch features are L2-normalised, so a linear head can
  only reach a narrow logit range; under logit adjustment the prior shift would
  then dominate. One scalar restores that freedom.
* Frozen BatchNorm statistics. With small batches and large subject shifts, the
  running statistics are a real source of instability, so the ImageNet ones are
  kept.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def _backbone(pretrained):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return resnet18(weights=weights)


def inflate_conv1(conv, in_channels, extra_scale=0.5):
    """Widen the first convolution to `in_channels` without weakening the
    pretrained RGB path: the original kernels are copied, the extra channels are
    tiled copies scaled down by `extra_scale`."""
    if in_channels == conv.in_channels:
        return conv
    new_conv = nn.Conv2d(in_channels, conv.out_channels,
                         kernel_size=conv.kernel_size, stride=conv.stride,
                         padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        weight = conv.weight.data
        base = weight.shape[1]
        new_conv.weight.zero_()
        new_conv.weight[:, :base].copy_(weight)
        if in_channels > base and extra_scale != 0.0:
            repeats = (in_channels - base + base - 1) // base
            tiled = weight.repeat(1, repeats, 1, 1)[:, : in_channels - base]
            new_conv.weight[:, base:].copy_(tiled * extra_scale)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data)
    return new_conv


class SharedStem(nn.Module):
    """conv1 .. layer2, shared by both regions."""

    def __init__(self, in_channels=6, pretrained=True, extra_scale=0.5, dropout=0.2):
        super().__init__()
        model = _backbone(pretrained)
        self.layers = nn.Sequential(
            inflate_conv1(model.conv1, in_channels, extra_scale),
            model.bn1, model.relu, model.maxpool,
            model.layer1, nn.Dropout(dropout),
            model.layer2, nn.Dropout(dropout),
        )
        del model

    def forward(self, x):
        return self.layers(x)


class RegionBranch(nn.Module):
    """layer3 .. layer4 plus pooling, one per region. Index 2 is layer4, which
    is the usual Grad-CAM target."""

    def __init__(self, pretrained=True, dropout=0.2, feat_norm=True):
        super().__init__()
        model = _backbone(pretrained)
        self.layers = nn.Sequential(
            model.layer3, nn.Dropout(dropout),
            model.layer4,
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1),
        )
        self.feat_norm = feat_norm
        del model

    def forward(self, x):
        out = self.layers(x)
        return F.normalize(out, dim=1) if self.feat_norm else out


class CARFNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=6, pretrained=True,
                 class_aware_fusion=True, joint_head=True, extra_scale=0.5,
                 dropout=0.2, head_dropout=0.3, logit_scale_init=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.base_layers = SharedStem(in_channels=in_channels, pretrained=pretrained,
                                      extra_scale=extra_scale, dropout=dropout)
        self.eyes_branch = RegionBranch(pretrained=pretrained, dropout=dropout)
        self.mouth_branch = RegionBranch(pretrained=pretrained, dropout=dropout)

        # Area Weighting Module
        self.area_weight = nn.Sequential(
            nn.Linear(512 * 2, 1024), nn.LayerNorm(1024), nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.Sigmoid(),
        )

        def head(dim):
            return nn.Sequential(
                nn.Linear(dim, 512), nn.LayerNorm(512), nn.ReLU(),
                nn.Dropout(head_dropout), nn.Linear(512, num_classes),
            )

        self.eyes = head(512)
        self.mouth = head(512)
        self.joint_head = joint_head
        self.joint = head(1024) if joint_head else None

        self.class_aware_fusion = class_aware_fusion
        self.fusion_gate = nn.Parameter(torch.zeros(num_classes))
        self.joint_gate = nn.Parameter(torch.zeros(1))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

    # ------------------------------------------------------------------ #
    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)

    def set_bn_frozen(self, flag=True):
        self._bn_frozen = flag
        if flag:
            self.freeze_bn()

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "_bn_frozen", False):
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    # ------------------------------------------------------------------ #
    def get_features(self, eyes, mouth):
        return (self.eyes_branch(self.base_layers(eyes)),
                self.mouth_branch(self.base_layers(mouth)))

    def fusing(self, eyes_features, mouth_features):
        combine = torch.cat([eyes_features, mouth_features], dim=-1)
        combine = combine * self.area_weight(combine.detach())
        contrast = F.normalize(combine, dim=-1)
        if self.training:
            contrast = F.dropout(contrast, 0.1)
        return combine[:, :512], combine[:, 512:], combine, contrast

    def _heads(self, eyes, mouth):
        eyes_features, mouth_features = self.get_features(eyes, mouth)
        eyes_out, mouth_out, combine, contrast = self.fusing(eyes_features, mouth_features)
        joint_logits = self.joint(combine) if self.joint_head else None
        return self.eyes(eyes_out), self.mouth(mouth_out), joint_logits, contrast

    def fuse_logits(self, eyes_logits, mouth_logits, joint_logits=None):
        if self.class_aware_fusion:
            gate = torch.sigmoid(self.fusion_gate).unsqueeze(0)
            fused = 2.0 * (gate * eyes_logits + (1.0 - gate) * mouth_logits)
        else:
            fused = eyes_logits + mouth_logits
        if joint_logits is not None:
            alpha = torch.sigmoid(self.joint_gate)
            fused = (1.0 - alpha) * fused + alpha * 2.0 * joint_logits
        return self.logit_scale * fused

    def forward(self, eyes, mouth):
        return self._heads(eyes, mouth)

    def predict(self, eyes, mouth, get_features=False):
        """Fused logits. This is the score to back-propagate for Grad-CAM."""
        if self.training:
            warnings.warn("predict() called while the model is in training mode",
                          RuntimeWarning)
        eyes_logits, mouth_logits, joint_logits, contrast = self._heads(eyes, mouth)
        fused = self.fuse_logits(eyes_logits, mouth_logits, joint_logits)
        return (fused, contrast) if get_features else fused

    # ------------------------------------------------------------------ #
    def param_groups(self, lr_backbone, lr_head, gate_mult=20.0, weight_decay=1e-4):
        """Layer-wise learning rates: a small one for the pretrained backbone, a
        larger one for the randomly initialised heads, and a much larger one
        without weight decay for the handful of gate scalars."""
        gate_names = {"fusion_gate", "joint_gate", "logit_scale"}
        backbone, heads, gates = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name in gate_names:
                gates.append(param)
            elif name.startswith(("base_layers", "eyes_branch", "mouth_branch")):
                backbone.append(param)
            else:
                heads.append(param)
        return [
            {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": heads, "lr": lr_head, "weight_decay": weight_decay},
            {"params": gates, "lr": lr_head * gate_mult, "weight_decay": 0.0},
        ]

    def gate_report(self, labels=None):
        gate = torch.sigmoid(self.fusion_gate).detach().cpu().tolist()
        extra = {"joint_alpha": round(float(torch.sigmoid(self.joint_gate).item()), 3),
                 "logit_scale": round(float(self.logit_scale.item()), 3)}
        if labels is None:
            return {"per_class": [round(g, 3) for g in gate], **extra}
        return {"per_class": {n: round(v, 3) for n, v in zip(labels, gate)}, **extra}
