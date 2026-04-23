import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) loading + fair cleaning
# ============================================================


def _safe_len(x):
    return len(x) if isinstance(x, np.ndarray) else 0


def _has_enough_finite(x, min_len=1, min_finite=10):
    return (
        isinstance(x, np.ndarray)
        and len(x) >= min_len
        and np.isfinite(np.asarray(x)).sum() >= min_finite
    )


def load_pickle_df(pkl_path="multimodal_trials.pkl"):
    return pd.read_pickle(pkl_path).copy()


def build_fair_comparison_df(
    df,
    min_pupil_len=1000,
    min_gaze_len=1000,
    max_gaze_len=20000,
    min_finite=10,
):
    pupil_len = df["pupil"].apply(_safe_len)
    gaze_len = df["gaze_x"].apply(_safe_len)

    keep = (
        (pupil_len >= min_pupil_len)
        & (gaze_len >= min_gaze_len)
        & (gaze_len <= max_gaze_len)
        & df["pupil"].apply(lambda x: _has_enough_finite(x, min_pupil_len, min_finite))
        & df["gaze_x"].apply(lambda x: _has_enough_finite(x, min_gaze_len, min_finite))
        & df["gaze_y"].apply(lambda x: _has_enough_finite(x, min_gaze_len, min_finite))
    )

    return df.loc[keep].copy().reset_index(drop=True)


def build_pupil_only_df(
    df,
    min_pupil_len=1000,
    min_finite=10,
):
    pupil_len = df["pupil"].apply(_safe_len)
    keep = (
        (pupil_len >= min_pupil_len)
        & df["pupil"].apply(lambda x: _has_enough_finite(x, min_pupil_len, min_finite))
    )
    return df.loc[keep].copy().reset_index(drop=True)


# ============================================================
# 2) subject-wise splits
# ============================================================


def split_subjectwise(
    df,
    val_frac=0.10,
    test_frac=0.20,
    seed=42,
):
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


# ============================================================
# 3) time-aware preprocessing helpers
# ============================================================


def _prepare_time_array(t, expected_len):
    if not isinstance(t, np.ndarray):
        return None
    t = np.asarray(t, dtype=np.float32).reshape(-1)
    if len(t) != expected_len:
        return None
    finite = np.isfinite(t)
    if finite.sum() < 2:
        return None
    t = t[finite]
    if len(t) < 2:
        return None
    if np.allclose(t.max(), t.min()):
        return None
    return t


def interpolate_nans_1d(x):
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
    return np.interp(new_grid, old_grid, x).astype(np.float32)


def resample_mask(mask, target_len, threshold=0.5):
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    rs = resample_1d(mask, target_len)
    return (rs >= threshold).astype(np.float32)


def time_resample_signal_and_mask(x, t, target_len):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    observed = np.isfinite(x).astype(np.float32)

    if not isinstance(t, np.ndarray) or len(t) != len(x):
        filled, observed_mask = interpolate_nans_1d(x)
        return resample_1d(zscore_1d(filled), target_len), resample_mask(observed_mask, target_len)

    t_arr = np.asarray(t, dtype=np.float32).reshape(-1)
    finite_t = np.isfinite(t_arr)
    if finite_t.sum() < 2:
        filled, observed_mask = interpolate_nans_1d(x)
        return resample_1d(zscore_1d(filled), target_len), resample_mask(observed_mask, target_len)

    x_valid = x[finite_t]
    t_valid = t_arr[finite_t]
    obs_valid = observed[finite_t]

    if len(t_valid) < 2 or np.allclose(t_valid.max(), t_valid.min()):
        filled, observed_mask = interpolate_nans_1d(x)
        return resample_1d(zscore_1d(filled), target_len), resample_mask(observed_mask, target_len)

    order = np.argsort(t_valid)
    t_valid = t_valid[order]
    x_valid = x_valid[order]
    obs_valid = obs_valid[order]

    t_unique, unique_idx = np.unique(t_valid, return_index=True)
    x_valid = x_valid[unique_idx]
    obs_valid = obs_valid[unique_idx]
    t_valid = t_unique

    good = np.isfinite(x_valid)
    if good.sum() == 0:
        return np.zeros((target_len,), dtype=np.float32), np.zeros((target_len,), dtype=np.float32)

    if good.sum() == 1:
        filled = np.full((target_len,), float(x_valid[good][0]), dtype=np.float32)
        obs_mask = np.zeros((target_len,), dtype=np.float32)
        # approximate observed region at the single point
        single_idx = target_len // 2
        obs_mask[single_idx] = 1.0
        return zscore_1d(filled), obs_mask

    new_t = np.linspace(t_valid.min(), t_valid.max(), target_len, dtype=np.float32)
    filled = np.interp(new_t, t_valid[good], x_valid[good]).astype(np.float32)
    obs_interp = np.interp(new_t, t_valid, obs_valid).astype(np.float32)
    obs_mask = (obs_interp >= 0.5).astype(np.float32)
    return zscore_1d(filled), obs_mask


def preprocess_pupil(pupil, pupil_time=None, target_len=512):
    filled, observed_mask = time_resample_signal_and_mask(pupil, pupil_time, target_len)
    return filled, observed_mask


def preprocess_gaze(gaze_x, gaze_y, gaze_time=None, target_len=512, add_velocity=True):
    gx, mx = time_resample_signal_and_mask(gaze_x, gaze_time, target_len)
    gy, my = time_resample_signal_and_mask(gaze_y, gaze_time, target_len)

    obs_mask = ((mx * my) >= 0.5).astype(np.float32)

    if add_velocity:
        dx = np.diff(gx, prepend=gx[0]).astype(np.float32)
        dy = np.diff(gy, prepend=gy[0]).astype(np.float32)
        feat = np.stack([gx, gy, dx, dy], axis=0).astype(np.float32)
    else:
        feat = np.stack([gx, gy], axis=0).astype(np.float32)

    return feat, obs_mask.astype(np.float32)


# ============================================================
# 4) metadata helpers
# ============================================================


def finite_fraction(x):
    if not isinstance(x, np.ndarray) or len(x) == 0:
        return 0.0
    x = np.asarray(x)
    return float(np.isfinite(x).mean())


def leading_missing_fraction(x):
    if not isinstance(x, np.ndarray) or len(x) == 0:
        return 1.0
    good = np.isfinite(np.asarray(x))
    if good.sum() == 0:
        return 1.0
    first_good = int(np.argmax(good))
    return float(first_good / len(good))


def trailing_missing_fraction(x):
    if not isinstance(x, np.ndarray) or len(x) == 0:
        return 1.0
    good = np.isfinite(np.asarray(x))
    if good.sum() == 0:
        return 1.0
    last_good = len(good) - 1 - int(np.argmax(good[::-1]))
    return float((len(good) - 1 - last_good) / len(good))


def _robust_mean_std(values):
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0, 1.0
    mu = float(np.mean(v))
    sigma = float(np.std(v))
    if not np.isfinite(sigma) or sigma < 1e-8:
        sigma = 1.0
    return mu, sigma


class MetadataSpec:
    def __init__(self, train_df: pd.DataFrame):
        distractors = sorted(train_df["distractor"].astype(str).fillna("UNK").unique().tolist())
        self.distractor_to_idx = {name: i for i, name in enumerate(distractors)}
        self.num_distractors = len(distractors)

        self.age_mean, self.age_std = _robust_mean_std(train_df["age"].to_numpy())
        self.load_mean, self.load_std = _robust_mean_std(train_df["load"].to_numpy())
        self.perform_mean, self.perform_std = _robust_mean_std(train_df["perform"].to_numpy())
        self.rtime_mean, self.rtime_std = _robust_mean_std(train_df["rtime"].to_numpy())

        pupil_lengths = train_df["pupil"].apply(_safe_len).to_numpy()
        gaze_lengths = train_df["gaze_x"].apply(_safe_len).to_numpy()
        self.pupil_loglen_mean, self.pupil_loglen_std = _robust_mean_std(np.log1p(pupil_lengths))
        self.gaze_loglen_mean, self.gaze_loglen_std = _robust_mean_std(np.log1p(gaze_lengths))

        self.meta_dim = 9 + self.num_distractors

    def encode_row(self, row: pd.Series):
        age = float(row.get("age", 0.0))
        load = float(row.get("load", 0.0))
        perform = float(row.get("perform", 0.0))
        rtime = row.get("rtime", np.nan)
        pupil = row.get("pupil", None)
        gaze_x = row.get("gaze_x", None)
        gaze_y = row.get("gaze_y", None)

        pupil_finite = finite_fraction(pupil)
        gaze_finite = min(finite_fraction(gaze_x), finite_fraction(gaze_y))
        pupil_lead = leading_missing_fraction(pupil)
        gaze_lead = max(leading_missing_fraction(gaze_x), leading_missing_fraction(gaze_y))
        pupil_trail = trailing_missing_fraction(pupil)
        gaze_trail = max(trailing_missing_fraction(gaze_x), trailing_missing_fraction(gaze_y))

        pupil_loglen = np.log1p(_safe_len(pupil))
        gaze_loglen = np.log1p(_safe_len(gaze_x))

        rtime_missing = 0.0 if np.isfinite(rtime) else 1.0
        rtime_filled = float(rtime) if np.isfinite(rtime) else self.rtime_mean

        distractor_onehot = np.zeros((self.num_distractors,), dtype=np.float32)
        dname = str(row.get("distractor", "UNK"))
        if dname in self.distractor_to_idx:
            distractor_onehot[self.distractor_to_idx[dname]] = 1.0

        features = np.array([
            (age - self.age_mean) / self.age_std,
            (load - self.load_mean) / self.load_std,
            (perform - self.perform_mean) / self.perform_std,
            (rtime_filled - self.rtime_mean) / self.rtime_std,
            rtime_missing,
            pupil_finite,
            gaze_finite,
            (pupil_loglen - self.pupil_loglen_mean) / self.pupil_loglen_std,
            (gaze_loglen - self.gaze_loglen_mean) / self.gaze_loglen_std,
            pupil_lead,
            gaze_lead,
            pupil_trail,
            gaze_trail,
        ], dtype=np.float32)

        # keep first 9 scalar task/quality features and append distractor one-hot + remaining quality features
        core = np.array([
            (age - self.age_mean) / self.age_std,
            (load - self.load_mean) / self.load_std,
            (perform - self.perform_mean) / self.perform_std,
            (rtime_filled - self.rtime_mean) / self.rtime_std,
            rtime_missing,
            pupil_finite,
            gaze_finite,
            (pupil_loglen - self.pupil_loglen_mean) / self.pupil_loglen_std,
            (gaze_loglen - self.gaze_loglen_mean) / self.gaze_loglen_std,
        ], dtype=np.float32)
        tail = np.array([pupil_lead, gaze_lead, pupil_trail, gaze_trail], dtype=np.float32)
        return np.concatenate([core, distractor_onehot, tail], axis=0).astype(np.float32)

    @property
    def dim(self):
        return 9 + self.num_distractors + 4


# ============================================================
# 5) datasets
# ============================================================


class ChoiDataset(Dataset):
    def __init__(self, df, pupil_len=512):
        self.df = df.reset_index(drop=True)
        self.pupil_len = pupil_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pupil, pupil_obs_mask = preprocess_pupil(
            row["pupil"],
            pupil_time=row.get("pupil_time", None),
            target_len=self.pupil_len,
        )

        return {
            "pupil": torch.tensor(pupil[None, :], dtype=torch.float32),
            "pupil_obs_mask": torch.tensor(pupil_obs_mask[None, :], dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "subject_id": torch.tensor(int(row["subject_id"]), dtype=torch.long),
            "trial": torch.tensor(int(row["trial"]), dtype=torch.long),
        }


class DengDataset(Dataset):
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
            gaze_time=row.get("gaze_time", None),
            target_len=self.gaze_len,
            add_velocity=self.add_velocity,
        )

        return {
            "gaze": torch.tensor(gaze, dtype=torch.float32),
            "gaze_obs_mask": torch.tensor(gaze_obs_mask[None, :], dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "subject_id": torch.tensor(int(row["subject_id"]), dtype=torch.long),
            "trial": torch.tensor(int(row["trial"]), dtype=torch.long),
        }


class FusionDataset(Dataset):
    def __init__(self, df, metadata_spec: MetadataSpec, pupil_len=512, gaze_len=512, add_velocity=True):
        self.df = df.reset_index(drop=True)
        self.metadata_spec = metadata_spec
        self.meta_dim = metadata_spec.dim
        self.pupil_len = pupil_len
        self.gaze_len = gaze_len
        self.add_velocity = add_velocity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pupil, pupil_obs_mask = preprocess_pupil(
            row["pupil"],
            pupil_time=row.get("pupil_time", None),
            target_len=self.pupil_len,
        )

        gaze, gaze_obs_mask = preprocess_gaze(
            row["gaze_x"],
            row["gaze_y"],
            gaze_time=row.get("gaze_time", None),
            target_len=self.gaze_len,
            add_velocity=self.add_velocity,
        )

        meta = self.metadata_spec.encode_row(row)

        return {
            "pupil": torch.tensor(pupil[None, :], dtype=torch.float32),
            "pupil_obs_mask": torch.tensor(pupil_obs_mask[None, :], dtype=torch.float32),
            "gaze": torch.tensor(gaze, dtype=torch.float32),
            "gaze_obs_mask": torch.tensor(gaze_obs_mask[None, :], dtype=torch.float32),
            "meta": torch.tensor(meta, dtype=torch.float32),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "subject_id": torch.tensor(int(row["subject_id"]), dtype=torch.long),
            "trial": torch.tensor(int(row["trial"]), dtype=torch.long),
        }


# ============================================================
# 6) dataloader builders
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
    metadata_spec = MetadataSpec(train_df)
    train_ds = FusionDataset(train_df, metadata_spec=metadata_spec, pupil_len=pupil_len, gaze_len=gaze_len, add_velocity=add_velocity)
    val_ds = FusionDataset(val_df, metadata_spec=metadata_spec, pupil_len=pupil_len, gaze_len=gaze_len, add_velocity=add_velocity)
    test_ds = FusionDataset(test_df, metadata_spec=metadata_spec, pupil_len=pupil_len, gaze_len=gaze_len, add_velocity=add_velocity)

    return {
        "train": make_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": make_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "meta_dim": metadata_spec.dim,
        "metadata_spec": metadata_spec,
    }


# ============================================================
# 7) convenience wrappers
# ============================================================


def build_all_loaders_for_fair_comparison(
    pkl_path="multimodal_trials.pkl",
    batch_size=32,
    pupil_len=512,
    gaze_len=512,
    seed=42,
    num_workers=0,
):
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
