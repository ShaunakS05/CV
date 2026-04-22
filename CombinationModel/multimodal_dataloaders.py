import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) loading + fair cleaning
# ============================================================

def _has_array(x, min_len=1):
    return isinstance(x, np.ndarray) and len(x) >= min_len


def _safe_len(x):
    return len(x) if isinstance(x, np.ndarray) else 0


def load_pickle_df(pkl_path="multimodal_trials.pkl"):
    df = pd.read_pickle(pkl_path).copy()
    return df


def build_fair_comparison_df(
    df,
    min_pupil_len=1000,
    min_gaze_len=1000,
    max_gaze_len=20000,
):
    """
    Fair comparison subset:
    - valid pupil
    - valid gaze
    - remove extreme gaze outliers
    This is the subset you should use for Choi vs Deng vs Fusion comparisons.
    """
    pupil_len = df["pupil"].apply(_safe_len)
    gaze_len = df["gaze_x"].apply(_safe_len)

    keep = (
        (pupil_len >= min_pupil_len)
        & (gaze_len >= min_gaze_len)
        & (gaze_len <= max_gaze_len)
        & df["pupil"].apply(lambda x: _has_array(x, min_pupil_len))
        & df["gaze_x"].apply(lambda x: _has_array(x, min_gaze_len))
        & df["gaze_y"].apply(lambda x: _has_array(x, min_gaze_len))
    )

    out = df.loc[keep].copy().reset_index(drop=True)
    return out


def build_pupil_only_df(
    df,
    min_pupil_len=1000,
):
    """
    Broader Choi-only subset if you want to use all usable pupil rows.
    Not ideal for apples-to-apples comparison against Deng/Fusion.
    """
    pupil_len = df["pupil"].apply(_safe_len)
    keep = (
        (pupil_len >= min_pupil_len)
        & df["pupil"].apply(lambda x: _has_array(x, min_pupil_len))
    )
    out = df.loc[keep].copy().reset_index(drop=True)
    return out


# ============================================================
# 2) subject-wise splits
# ============================================================

def split_subjectwise(
    df,
    val_frac=0.10,
    test_frac=0.20,
    seed=42,
):
    """
    Single subject-wise split.
    Same split should be reused for Choi/Deng/Fusion.
    """
    subjects = np.array(sorted(df["subject_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_test = max(1, int(round(test_frac * n_subjects)))
    n_val = max(1, int(round(val_frac * n_subjects)))

    test_subjects = set(subjects[:n_test].tolist())
    val_subjects = set(subjects[n_test:n_test + n_val].tolist())
    train_subjects = set(subjects[n_test + n_val:].tolist())

    train_df = df[df["subject_id"].isin(train_subjects)].copy().reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjects)].copy().reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjects)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def loso_splits(df):
    """
    Leave-one-subject-out generator.
    Good for final evaluation.
    """
    subjects = sorted(df["subject_id"].unique())
    for heldout in subjects:
        test_df = df[df["subject_id"] == heldout].copy().reset_index(drop=True)
        train_df = df[df["subject_id"] != heldout].copy().reset_index(drop=True)
        yield heldout, train_df, test_df


# ============================================================
# 3) preprocessing helpers
# ============================================================

def interpolate_nans_1d(x):
    """
    Returns:
      filled: float32 array
      observed_mask: 1 where original value was finite, else 0
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    observed = np.isfinite(x).astype(np.float32)

    if observed.sum() == 0:
        return np.zeros_like(x, dtype=np.float32), observed

    idx = np.arange(len(x), dtype=np.float32)
    good = np.isfinite(x)

    if good.sum() == 1:
        filled = np.full_like(x, fill_value=x[good][0], dtype=np.float32)
        return filled, observed

    filled = np.interp(idx, idx[good], x[good]).astype(np.float32)
    return filled, observed


def zscore_1d(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    mu = np.nanmean(x)
    sigma = np.nanstd(x)
    if not np.isfinite(sigma) or sigma < 1e-8:
        return (x - mu).astype(np.float32)
    return ((x - mu) / sigma).astype(np.float32)


def resample_1d(x, target_len):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == target_len:
        return x.astype(np.float32)
    if len(x) < 2:
        return np.full((target_len,), x[0] if len(x) == 1 else 0.0, dtype=np.float32)

    old_grid = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.interp(new_grid, old_grid, x).astype(np.float32)
    return out


def resample_mask(mask, target_len, threshold=0.5):
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    rs = resample_1d(mask, target_len)
    return (rs >= threshold).astype(np.float32)


def preprocess_pupil(pupil, target_len=512):
    """
    Choi-friendly processing:
    - keep original missingness mask
    - interpolate NaNs for network input
    - z-score per trial
    - resample to fixed length
    """
    filled, observed_mask = interpolate_nans_1d(pupil)
    filled = zscore_1d(filled)

    filled = resample_1d(filled, target_len)
    observed_mask = resample_mask(observed_mask, target_len)

    return filled, observed_mask


def preprocess_gaze(gaze_x, gaze_y, target_len=512, add_velocity=True):
    """
    Deng-inspired processing:
    - interpolate NaNs in x/y
    - z-score x and y per trial
    - resample
    - optionally add dx/dy channels
    """
    gx, mx = interpolate_nans_1d(gaze_x)
    gy, my = interpolate_nans_1d(gaze_y)

    gx = zscore_1d(gx)
    gy = zscore_1d(gy)

    gx = resample_1d(gx, target_len)
    gy = resample_1d(gy, target_len)

    obs_mask = resample_mask((mx * my).astype(np.float32), target_len)

    if add_velocity:
        dx = np.diff(gx, prepend=gx[0]).astype(np.float32)
        dy = np.diff(gy, prepend=gy[0]).astype(np.float32)
        feat = np.stack([gx, gy, dx, dy], axis=0).astype(np.float32)
    else:
        feat = np.stack([gx, gy], axis=0).astype(np.float32)

    return feat, obs_mask.astype(np.float32)


# ============================================================
# 4) datasets
# ============================================================

class ChoiDataset(Dataset):
    """
    Pupil-only dataset.
    Useful for a Choi-style model with classification + optional imputation.
    """
    def __init__(self, df, pupil_len=512):
        self.df = df.reset_index(drop=True)
        self.pupil_len = pupil_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pupil, pupil_obs_mask = preprocess_pupil(
            row["pupil"],
            target_len=self.pupil_len,
        )

        return {
            "pupil": torch.tensor(pupil[None, :], dtype=torch.float32),           # (1, T)
            "pupil_obs_mask": torch.tensor(pupil_obs_mask[None, :], dtype=torch.float32),  # (1, T)
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "subject_id": torch.tensor(int(row["subject_id"]), dtype=torch.long),
            "trial": torch.tensor(int(row["trial"]), dtype=torch.long),
        }


class DengDataset(Dataset):
    """
    Gaze-only dataset.
    Deng-inspired, but without saliency/video context.
    """
    def __init__(self, df, gaze_len=512, add_velocity=True):
        self.df = df.reset_index(drop=True)
        self.gaze_len = gaze_len
        self.add_velocity = add_velocity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        gaze, gaze_obs_mask = preprocess_gaze(
            row["gaze_x"],
            row["gaze_y"],
            target_len=self.gaze_len,
            add_velocity=self.add_velocity,
        )

        return {
            "gaze": torch.tensor(gaze, dtype=torch.float32),                     # (C, T)
            "gaze_obs_mask": torch.tensor(gaze_obs_mask[None, :], dtype=torch.float32),  # (1, T)
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "subject_id": torch.tensor(int(row["subject_id"]), dtype=torch.long),
            "trial": torch.tensor(int(row["trial"]), dtype=torch.long),
        }


class FusionDataset(Dataset):
    """
    Pupil + gaze dataset for the proposed fusion model.
    """
    def __init__(self, df, pupil_len=512, gaze_len=512, add_velocity=True):
        self.df = df.reset_index(drop=True)
        self.pupil_len = pupil_len
        self.gaze_len = gaze_len
        self.add_velocity = add_velocity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pupil, pupil_obs_mask = preprocess_pupil(
            row["pupil"],
            target_len=self.pupil_len,
        )

        gaze, gaze_obs_mask = preprocess_gaze(
            row["gaze_x"],
            row["gaze_y"],
            target_len=self.gaze_len,
            add_velocity=self.add_velocity,
        )

        return {
            "pupil": torch.tensor(pupil[None, :], dtype=torch.float32),          # (1, Tp)
            "pupil_obs_mask": torch.tensor(pupil_obs_mask[None, :], dtype=torch.float32),
            "gaze": torch.tensor(gaze, dtype=torch.float32),                      # (Cg, Tg)
            "gaze_obs_mask": torch.tensor(gaze_obs_mask[None, :], dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "subject_id": torch.tensor(int(row["subject_id"]), dtype=torch.long),
            "trial": torch.tensor(int(row["trial"]), dtype=torch.long),
        }


# ============================================================
# 5) dataloader builders
# ============================================================

def make_loader(dataset, batch_size=32, shuffle=False, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_choi_loaders(
    train_df,
    val_df,
    test_df,
    batch_size=32,
    pupil_len=512,
    num_workers=0,
):
    train_ds = ChoiDataset(train_df, pupil_len=pupil_len)
    val_ds = ChoiDataset(val_df, pupil_len=pupil_len)
    test_ds = ChoiDataset(test_df, pupil_len=pupil_len)

    return {
        "train": make_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def make_deng_loaders(
    train_df,
    val_df,
    test_df,
    batch_size=32,
    gaze_len=512,
    add_velocity=True,
    num_workers=0,
):
    train_ds = DengDataset(train_df, gaze_len=gaze_len, add_velocity=add_velocity)
    val_ds = DengDataset(val_df, gaze_len=gaze_len, add_velocity=add_velocity)
    test_ds = DengDataset(test_df, gaze_len=gaze_len, add_velocity=add_velocity)

    return {
        "train": make_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def make_fusion_loaders(
    train_df,
    val_df,
    test_df,
    batch_size=32,
    pupil_len=512,
    gaze_len=512,
    add_velocity=True,
    num_workers=0,
):
    train_ds = FusionDataset(train_df, pupil_len=pupil_len, gaze_len=gaze_len, add_velocity=add_velocity)
    val_ds = FusionDataset(val_df, pupil_len=pupil_len, gaze_len=gaze_len, add_velocity=add_velocity)
    test_ds = FusionDataset(test_df, pupil_len=pupil_len, gaze_len=gaze_len, add_velocity=add_velocity)

    return {
        "train": make_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


# ============================================================
# 6) convenience wrappers
# ============================================================

def build_all_loaders_for_fair_comparison(
    pkl_path="multimodal_trials.pkl",
    batch_size=32,
    pupil_len=512,
    gaze_len=512,
    seed=42,
    num_workers=0,
):
    """
    This is the one to use for fair Choi vs Deng vs Fusion comparison.
    All 3 loaders are built from the exact same cleaned subset and same subject split.
    """
    df = load_pickle_df(pkl_path)
    fair_df = build_fair_comparison_df(df)

    train_df, val_df, test_df = split_subjectwise(fair_df, seed=seed)

    choi = make_choi_loaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        pupil_len=pupil_len,
        num_workers=num_workers,
    )

    deng = make_deng_loaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        gaze_len=gaze_len,
        add_velocity=True,
        num_workers=num_workers,
    )

    fusion = make_fusion_loaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        pupil_len=pupil_len,
        gaze_len=gaze_len,
        add_velocity=True,
        num_workers=num_workers,
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "choi": choi,
        "deng": deng,
        "fusion": fusion,
    }


def build_choi_only_loaders(
    pkl_path="multimodal_trials.pkl",
    batch_size=32,
    pupil_len=512,
    seed=42,
    num_workers=0,
    use_all_pupil_rows=True,
):
    """
    Use this only if you want a stronger standalone Choi baseline.
    For fair comparison against Deng/Fusion, use build_all_loaders_for_fair_comparison instead.
    """
    df = load_pickle_df(pkl_path)
    if use_all_pupil_rows:
        df = build_pupil_only_df(df)
    else:
        df = build_fair_comparison_df(df)

    train_df, val_df, test_df = split_subjectwise(df, seed=seed)

    choi = make_choi_loaders(
        train_df, val_df, test_df,
        batch_size=batch_size,
        pupil_len=pupil_len,
        num_workers=num_workers,
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "choi": choi,
    }


# ============================================================
# 7) quick test
# ============================================================

if __name__ == "__main__":
    bundle = build_all_loaders_for_fair_comparison(
        pkl_path="multimodal_trials.pkl",
        batch_size=8,
        pupil_len=512,
        gaze_len=512,
        seed=42,
        num_workers=0,
    )

    print("Train/Val/Test sizes:")
    print(bundle["train_df"].shape, bundle["val_df"].shape, bundle["test_df"].shape)

    batch_choi = next(iter(bundle["choi"]["train"]))
    print("\nChoi batch:")
    print(batch_choi["pupil"].shape, batch_choi["pupil_obs_mask"].shape, batch_choi["label"].shape)

    batch_deng = next(iter(bundle["deng"]["train"]))
    print("\nDeng batch:")
    print(batch_deng["gaze"].shape, batch_deng["gaze_obs_mask"].shape, batch_deng["label"].shape)

    batch_fusion = next(iter(bundle["fusion"]["train"]))
    print("\nFusion batch:")
    print(
        batch_fusion["pupil"].shape,
        batch_fusion["gaze"].shape,
        batch_fusion["label"].shape,
    )