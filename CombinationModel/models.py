import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# helpers
# ============================================================


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1, eps: float = 1e-8):
    if mask is None:
        return x.mean(dim=dim)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(eps)
    return num / den


def masked_global_avg_pool_1d(x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    if mask is None:
        return x.mean(dim=-1)
    if mask.shape[1] == 1 and x.shape[1] != 1:
        mask = mask.expand(-1, x.shape[1], -1)
    num = (x * mask).sum(dim=-1)
    den = mask.sum(dim=-1).clamp_min(1e-8)
    return num / den


def make_random_keep_mask(obs_mask: torch.Tensor, keep_prob: float = 0.90):
    rand = torch.rand_like(obs_mask)
    keep = ((rand < keep_prob).float() * obs_mask).float()

    B, _, _ = keep.shape
    for b in range(B):
        if obs_mask[b].sum() > 0 and keep[b].sum() == 0:
            first_idx = torch.nonzero(obs_mask[b, 0] > 0.5, as_tuple=False)[0, 0]
            keep[b, 0, first_idx] = 1.0

    return keep


def resize_sequence_and_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    target_len: int,
    mode: str = "linear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[-1] == target_len:
        return x, mask

    x_resized = F.interpolate(x, size=target_len, mode=mode, align_corners=False)
    mask_resized = F.interpolate(mask.float(), size=target_len, mode="nearest")
    mask_resized = (mask_resized > 0.5).float()
    return x_resized, mask_resized


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


class TemporalFusionBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.block = ConvBlock1D(d_model, d_model, kernel_size=kernel_size, dilation=1, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(y)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, expansion: int = 4):
        super().__init__()
        hidden = d_model * expansion
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class PupilDominantCrossBlock(nn.Module):
    """
    Pupil-dominant interaction:
    pupil attends to gaze, while gaze stays as a contextual encoder output.
    This matches the observed data where pupil seems to carry the stronger signal.
    """

    def __init__(self, d_model: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pupil_to_gaze = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(d_model)
        self.ff_p = FeedForwardBlock(d_model, dropout=dropout)

    def forward(
        self,
        p: torch.Tensor,
        g: torch.Tensor,
        g_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p_from_g, attn_pg = self.pupil_to_gaze(
            query=p,
            key=g,
            value=g,
            key_padding_mask=g_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        p = self.norm_p(p + p_from_g)
        p = self.ff_p(p)
        return p, attn_pg


# ============================================================
# 1) Choi-style model
# ============================================================


class ChoiPupilNet(nn.Module):
    def __init__(
        self,
        hidden_channels=(64, 128, 128),
        kernel_size: int = 5,
        dropout: float = 0.1,
        classifier_hidden: int = 128,
    ):
        super().__init__()

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
        x = torch.cat([pupil, pupil_obs_mask], dim=1)
        feat = self.encoder(x)

        recon = self.recon_head(feat)
        pooled = masked_global_avg_pool_1d(feat, pupil_obs_mask)
        logits = self.cls_head(pooled).squeeze(-1)

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
    pos_weight: Optional[torch.Tensor] = None,
):
    logits = outputs["logits"]
    recon = outputs["recon"]

    pupil = batch["pupil"]
    obs_mask = batch["pupil_obs_mask"]
    labels = batch["label"]

    cls_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

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
            in_ch=in_channels + 1,
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
        x = torch.cat([gaze, gaze_obs_mask], dim=1)
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
    pos_weight: Optional[torch.Tensor] = None,
):
    logits = outputs["logits"]
    labels = batch["label"]
    cls_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    return {
        "loss": cls_loss,
        "cls_loss": cls_loss.detach(),
    }


# ============================================================
# 3) Best-next hybrid multimodal model with metadata branch
# ============================================================


class FusionADHDNet(nn.Module):
    def __init__(
        self,
        meta_dim: int,
        pupil_hidden=(64, 128, 128),
        gaze_hidden=(64, 128, 128),
        kernel_size: int = 5,
        dropout: float = 0.15,
        gaze_in_channels: int = 4,
        d_model: int = 64,
        num_heads: int = 2,
        num_cross_blocks: int = 1,
        fusion_len: int = 192,
        classifier_hidden: int = 128,
        add_positional_encoding: bool = True,
    ):
        super().__init__()

        self.fusion_len = fusion_len
        self.meta_dim = meta_dim

        self.pupil_encoder = TCNEncoder(
            in_ch=2,
            channels=pupil_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.gaze_encoder = TCNEncoder(
            in_ch=gaze_in_channels + 1,
            channels=gaze_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        cp = self.pupil_encoder.out_channels
        cg = self.gaze_encoder.out_channels

        self.pupil_proj = nn.Conv1d(cp, d_model, kernel_size=1)
        self.gaze_proj = nn.Conv1d(cg, d_model, kernel_size=1)

        self.pos_enc = PositionalEncoding1D(d_model, max_len=max(4096, fusion_len)) if add_positional_encoding else nn.Identity()

        self.cross_blocks = nn.ModuleList(
            [PupilDominantCrossBlock(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(num_cross_blocks)]
        )

        self.joint_reduce = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_fusion = TemporalFusionBlock(d_model=d_model, kernel_size=kernel_size, dropout=dropout)

        self.unimodal_summary = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.hybrid_gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.meta_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.pupil_recon_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, 1, kernel_size=1),
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    def _make_key_padding_mask(self, obs_mask: torch.Tensor) -> torch.Tensor:
        return ~(obs_mask.squeeze(1) > 0.5)

    def _fix_all_masked_sequence(
        self,
        seq: torch.Tensor,
        mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_masked = key_padding_mask.all(dim=1)

        if all_masked.any():
            seq = seq.clone()
            mask = mask.clone()
            key_padding_mask = key_padding_mask.clone()

            seq[all_masked, 0, :] = 0.0
            mask[all_masked, 0, 0] = 1.0
            key_padding_mask[all_masked, 0] = False

        return seq, mask, key_padding_mask

    def forward(
        self,
        pupil: torch.Tensor,
        pupil_obs_mask: torch.Tensor,
        gaze: torch.Tensor,
        gaze_obs_mask: torch.Tensor,
        meta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p_in = torch.cat([pupil, pupil_obs_mask], dim=1)
        g_in = torch.cat([gaze, gaze_obs_mask], dim=1)

        p_feat = self.pupil_encoder(p_in)
        g_feat = self.gaze_encoder(g_in)

        p_feat = self.pupil_proj(p_feat)
        g_feat = self.gaze_proj(g_feat)

        p_feat, p_mask = resize_sequence_and_mask(p_feat, pupil_obs_mask, self.fusion_len)
        g_feat, g_mask = resize_sequence_and_mask(g_feat, gaze_obs_mask, self.fusion_len)

        p_seq = self.pos_enc(p_feat.transpose(1, 2))
        g_seq = self.pos_enc(g_feat.transpose(1, 2))

        p_seq = torch.nan_to_num(p_seq, nan=0.0, posinf=0.0, neginf=0.0)
        g_seq = torch.nan_to_num(g_seq, nan=0.0, posinf=0.0, neginf=0.0)

        g_key_padding_mask = self._make_key_padding_mask(g_mask)
        p_key_padding_mask = self._make_key_padding_mask(p_mask)

        p_seq, p_mask, p_key_padding_mask = self._fix_all_masked_sequence(
            p_seq, p_mask, p_key_padding_mask
        )
        g_seq, g_mask, g_key_padding_mask = self._fix_all_masked_sequence(
            g_seq, g_mask, g_key_padding_mask
        )

        attn_pg_all = []
        for block in self.cross_blocks:
            p_seq, attn_pg = block(
                p_seq,
                g_seq,
                g_key_padding_mask=g_key_padding_mask,
            )
            p_seq = torch.nan_to_num(p_seq, nan=0.0, posinf=0.0, neginf=0.0)
            attn_pg_all.append(attn_pg)

        joint_seq = self.joint_reduce(torch.cat([p_seq, g_seq], dim=-1))
        joint_seq = self.temporal_fusion(joint_seq)
        joint_seq = torch.nan_to_num(joint_seq, nan=0.0, posinf=0.0, neginf=0.0)

        joint_mask = (p_mask * g_mask).float()
        joint_mask = torch.where(
            joint_mask.sum(dim=-1, keepdim=True) > 0,
            joint_mask,
            torch.maximum(p_mask, g_mask),
        )

        p_interaction_feat = p_seq.transpose(1, 2)
        pupil_recon_sync = self.pupil_recon_head(p_interaction_feat)
        pupil_recon = F.interpolate(
            pupil_recon_sync,
            size=pupil.shape[-1],
            mode="linear",
            align_corners=False,
        )
        pupil_recon = torch.nan_to_num(pupil_recon, nan=0.0, posinf=0.0, neginf=0.0)

        joint_feat = joint_seq.transpose(1, 2)
        joint_pooled = masked_global_avg_pool_1d(joint_feat, joint_mask)
        p_summary = masked_global_avg_pool_1d(p_feat, p_mask)
        g_summary = masked_global_avg_pool_1d(g_feat, g_mask)
        residual_summary = self.unimodal_summary(torch.cat([p_summary, g_summary], dim=-1))

        gate = self.hybrid_gate(torch.cat([joint_pooled, residual_summary, p_summary], dim=-1))
        fused = gate * joint_pooled + (1.0 - gate) * residual_summary

        meta_emb = self.meta_encoder(meta)
        meta_gate = self.meta_gate(torch.cat([fused, meta_emb], dim=-1))
        pooled = meta_gate * fused + (1.0 - meta_gate) * meta_emb

        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.cls_head(pooled).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "logits": logits,
            "pupil_recon": pupil_recon,
            "pupil_feat": p_feat,
            "gaze_feat": g_feat,
            "pupil_seq_after_cross": p_seq,
            "gaze_seq_context": g_seq,
            "joint_seq": joint_seq,
            "joint_pooled": joint_pooled,
            "residual_summary": residual_summary,
            "meta_emb": meta_emb,
            "hybrid_gate": gate,
            "meta_gate": meta_gate,
            "pooled": pooled,
            "attn_pupil_to_gaze": attn_pg_all,
            "joint_mask": joint_mask,
        }


def fusion_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    recon_weight: float = 0.08,
    pos_weight: Optional[torch.Tensor] = None,
):
    logits = outputs["logits"]
    pupil_recon = outputs["pupil_recon"]

    labels = batch["label"]
    target_pupil = batch.get("pupil_recon_target", batch["pupil"])
    recon_mask = batch.get("pupil_recon_target_mask", batch["pupil_obs_mask"])

    cls_loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    recon_err = (pupil_recon - target_pupil) ** 2
    recon_loss = masked_mean(recon_err, recon_mask, dim=-1).mean()

    total = cls_loss + recon_weight * recon_loss
    return {
        "loss": total,
        "cls_loss": cls_loss.detach(),
        "recon_loss": recon_loss.detach(),
    }


# ============================================================
# 4) masked-imputation helper for the joint model
# ============================================================


def apply_training_mask_to_pupil(pupil: torch.Tensor, obs_mask: torch.Tensor, keep_prob: float = 0.92):
    keep_mask = make_random_keep_mask(obs_mask, keep_prob=keep_prob)
    target_mask = (obs_mask - keep_mask).clamp_min(0.0)
    masked_pupil = pupil * keep_mask
    return masked_pupil, keep_mask, target_mask


def prepare_masked_fusion_batch(batch: Dict[str, torch.Tensor], keep_prob: float = 0.92) -> Dict[str, torch.Tensor]:
    pupil = batch["pupil"]
    obs_mask = batch["pupil_obs_mask"]

    masked_pupil, keep_mask, target_mask = apply_training_mask_to_pupil(pupil, obs_mask, keep_prob=keep_prob)

    out = dict(batch)
    out["pupil_recon_target"] = pupil
    out["pupil_recon_target_mask"] = target_mask
    out["pupil"] = masked_pupil
    out["pupil_obs_mask"] = keep_mask
    return out
