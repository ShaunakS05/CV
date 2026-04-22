import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# helpers
# ============================================================

def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1, eps: float = 1e-8):
    """
    x:    (..., T)
    mask: same broadcastable shape as x, values in {0,1}
    """
    if mask is None:
        return x.mean(dim=dim)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(eps)
    return num / den


def masked_global_avg_pool_1d(x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    x:    (B, C, T)
    mask: (B, 1, T) or (B, C, T)
    """
    if mask is None:
        return x.mean(dim=-1)
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand(-1, x.shape[1], -1)
    num = (x * mask).sum(dim=-1)
    den = mask.sum(dim=-1).clamp_min(1e-8)
    return num / den


def make_random_keep_mask(obs_mask: torch.Tensor, keep_prob: float = 0.85):
    """
    obs_mask: (B, 1, T), 1 where originally observed
    returns keep mask over observed points
    """
    rand = torch.rand_like(obs_mask)
    keep = (rand < keep_prob).float()
    return keep * obs_mask


# ============================================================
# temporal building blocks
# ============================================================

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = ((kernel_size - 1) * dilation) // 2

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class TCNEncoder(nn.Module):
    def __init__(self, in_ch: int, channels=(64, 128, 128), kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = in_ch
        for i, ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(ConvBlock1D(prev, ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            prev = ch
        self.net = nn.Sequential(*layers)
        self.out_channels = prev

    def forward(self, x):
        return self.net(x)


# ============================================================
# 1) Choi-style model
# ============================================================

class ChoiPupilNet(nn.Module):
    """
    Input expected from ChoiDataset:
      pupil           : (B, 1, T)
      pupil_obs_mask  : (B, 1, T)

    Idea:
    - encode pupil + mask
    - reconstruct pupil sequence
    - classify ADHD
    """
    def __init__(
        self,
        hidden_channels=(64, 128, 128),
        kernel_size: int = 5,
        dropout: float = 0.1,
        classifier_hidden: int = 128,
    ):
        super().__init__()

        # channels: [pupil, observed_mask]
        self.encoder = TCNEncoder(
            in_ch=2,
            channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        c = self.encoder.out_channels

        self.recon_head = nn.Sequential(
            nn.Conv1d(c, c // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(c // 2, 1, kernel_size=1),
        )

        self.cls_head = nn.Sequential(
            nn.Linear(c, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, pupil: torch.Tensor, pupil_obs_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([pupil, pupil_obs_mask], dim=1)  # (B,2,T)
        feat = self.encoder(x)                         # (B,C,T)

        recon = self.recon_head(feat)                 # (B,1,T)
        pooled = masked_global_avg_pool_1d(feat, pupil_obs_mask)
        logits = self.cls_head(pooled).squeeze(-1)    # (B,)

        return {
            "logits": logits,
            "recon": recon,
            "feat": feat,
            "pooled": pooled,
        }


def choi_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    recon_weight: float = 1.0,
):
    """
    Classification + reconstruction on originally observed points.
    """
    logits = outputs["logits"]
    recon = outputs["recon"]

    pupil = batch["pupil"]
    obs_mask = batch["pupil_obs_mask"]
    labels = batch["label"]

    cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

    # reconstruction only where original values were observed
    recon_err = (recon - pupil) ** 2
    recon_loss = masked_mean(recon_err, obs_mask, dim=-1).mean()

    total = cls_loss + recon_weight * recon_loss
    return {
        "loss": total,
        "cls_loss": cls_loss.detach(),
        "recon_loss": recon_loss.detach(),
    }


# ============================================================
# 2) Deng-style model
# ============================================================

class DengGazeNet(nn.Module):
    """
    Input expected from DengDataset:
      gaze           : (B, C, T) where C is 4 if [x,y,dx,dy]
      gaze_obs_mask  : (B, 1, T)

    This is Deng-inspired for your dataset.
    It does NOT include stimulus saliency/video context.
    """
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels=(64, 128, 128),
        kernel_size: int = 5,
        dropout: float = 0.1,
        classifier_hidden: int = 128,
    ):
        super().__init__()

        self.encoder = TCNEncoder(
            in_ch=in_channels + 1,   # add observation mask as channel
            channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        c = self.encoder.out_channels

        self.cls_head = nn.Sequential(
            nn.Linear(c, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, gaze: torch.Tensor, gaze_obs_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_ch = gaze_obs_mask.expand(-1, 1, gaze.shape[-1])
        x = torch.cat([gaze, mask_ch], dim=1)
        feat = self.encoder(x)
        pooled = masked_global_avg_pool_1d(feat, gaze_obs_mask)
        logits = self.cls_head(pooled).squeeze(-1)

        return {
            "logits": logits,
            "feat": feat,
            "pooled": pooled,
        }


def deng_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
):
    logits = outputs["logits"]
    labels = batch["label"]
    cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
    return {
        "loss": cls_loss,
        "cls_loss": cls_loss.detach(),
    }


# ============================================================
# 3) Fusion model
# ============================================================

class GatedFusion(nn.Module):
    def __init__(self, dim_p: int, dim_g: int, fused_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj_p = nn.Linear(dim_p, fused_dim)
        self.proj_g = nn.Linear(dim_g, fused_dim)

        self.gate = nn.Sequential(
            nn.Linear(dim_p + dim_g, fused_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_p: torch.Tensor, z_g: torch.Tensor):
        p = self.proj_p(z_p)
        g = self.proj_g(z_g)
        alpha = self.gate(torch.cat([z_p, z_g], dim=-1))
        fused = alpha * p + (1.0 - alpha) * g
        return fused, alpha


class FusionADHDNet(nn.Module):
    """
    Input expected from FusionDataset:
      pupil, pupil_obs_mask, gaze, gaze_obs_mask

    Pupil branch keeps the Choi-style auxiliary reconstruction head.
    Gaze branch is Deng-inspired.
    Fusion is late gated fusion.
    """
    def __init__(
        self,
        pupil_hidden=(64, 128, 128),
        gaze_hidden=(64, 128, 128),
        kernel_size: int = 5,
        dropout: float = 0.1,
        gaze_in_channels: int = 4,
        fused_dim: int = 128,
        classifier_hidden: int = 128,
    ):
        super().__init__()

        # pupil branch
        self.pupil_encoder = TCNEncoder(
            in_ch=2,
            channels=pupil_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        cp = self.pupil_encoder.out_channels

        self.pupil_recon_head = nn.Sequential(
            nn.Conv1d(cp, cp // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cp // 2, 1, kernel_size=1),
        )

        # gaze branch
        self.gaze_encoder = TCNEncoder(
            in_ch=gaze_in_channels + 1,
            channels=gaze_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        cg = self.gaze_encoder.out_channels

        self.fusion = GatedFusion(cp, cg, fused_dim=fused_dim, dropout=dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(
        self,
        pupil: torch.Tensor,
        pupil_obs_mask: torch.Tensor,
        gaze: torch.Tensor,
        gaze_obs_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # pupil branch
        p_in = torch.cat([pupil, pupil_obs_mask], dim=1)
        p_feat = self.pupil_encoder(p_in)
        p_recon = self.pupil_recon_head(p_feat)
        z_p = masked_global_avg_pool_1d(p_feat, pupil_obs_mask)

        # gaze branch
        g_in = torch.cat([gaze, gaze_obs_mask.expand(-1, 1, gaze.shape[-1])], dim=1)
        g_feat = self.gaze_encoder(g_in)
        z_g = masked_global_avg_pool_1d(g_feat, gaze_obs_mask)

        z_fused, alpha = self.fusion(z_p, z_g)
        logits = self.cls_head(z_fused).squeeze(-1)

        return {
            "logits": logits,
            "pupil_recon": p_recon,
            "pupil_feat": p_feat,
            "gaze_feat": g_feat,
            "z_p": z_p,
            "z_g": z_g,
            "z_fused": z_fused,
            "fusion_gate": alpha,
        }


def fusion_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    recon_weight: float = 0.5,
):
    logits = outputs["logits"]
    pupil_recon = outputs["pupil_recon"]

    labels = batch["label"]
    pupil = batch["pupil"]
    pupil_obs_mask = batch["pupil_obs_mask"]

    cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

    recon_err = (pupil_recon - pupil) ** 2
    recon_loss = masked_mean(recon_err, pupil_obs_mask, dim=-1).mean()

    total = cls_loss + recon_weight * recon_loss
    return {
        "loss": total,
        "cls_loss": cls_loss.detach(),
        "recon_loss": recon_loss.detach(),
    }


# ============================================================
# 4) optional masked-imputation training helper
# ============================================================

def apply_training_mask_to_pupil(pupil: torch.Tensor, obs_mask: torch.Tensor, keep_prob: float = 0.85):
    """
    For a stronger Choi-style setup, randomly hide some originally observed points.
    Returns:
      masked_pupil_input
      keep_mask          (what remains visible)
      target_mask        (what was hidden from visible observed points)
    """
    keep_mask = make_random_keep_mask(obs_mask, keep_prob=keep_prob)
    target_mask = (obs_mask - keep_mask).clamp_min(0.0)
    masked_pupil = pupil * keep_mask
    return masked_pupil, keep_mask, target_mask


# ============================================================
# 5) smoke test
# ============================================================

if __name__ == "__main__":
    B, Tp, Tg = 8, 512, 512

    pupil = torch.randn(B, 1, Tp)
    pupil_mask = torch.ones(B, 1, Tp)

    gaze = torch.randn(B, 4, Tg)
    gaze_mask = torch.ones(B, 1, Tg)

    labels = torch.randint(0, 2, (B,)).float()

    batch_choi = {
        "pupil": pupil,
        "pupil_obs_mask": pupil_mask,
        "label": labels,
    }

    batch_deng = {
        "gaze": gaze,
        "gaze_obs_mask": gaze_mask,
        "label": labels,
    }

    batch_fusion = {
        "pupil": pupil,
        "pupil_obs_mask": pupil_mask,
        "gaze": gaze,
        "gaze_obs_mask": gaze_mask,
        "label": labels,
    }

    m1 = ChoiPupilNet()
    o1 = m1(batch_choi["pupil"], batch_choi["pupil_obs_mask"])
    print("Choi logits:", o1["logits"].shape, "recon:", o1["recon"].shape)

    m2 = DengGazeNet()
    o2 = m2(batch_deng["gaze"], batch_deng["gaze_obs_mask"])
    print("Deng logits:", o2["logits"].shape)

    m3 = FusionADHDNet()
    o3 = m3(
        batch_fusion["pupil"],
        batch_fusion["pupil_obs_mask"],
        batch_fusion["gaze"],
        batch_fusion["gaze_obs_mask"],
    )
    print("Fusion logits:", o3["logits"].shape, "pupil_recon:", o3["pupil_recon"].shape)