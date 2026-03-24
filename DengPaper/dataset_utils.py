# dataset_utils.py
import numpy as np
import torch
from torch.utils.data import Dataset

def pad_or_truncate(x, max_len):
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= max_len:
        return x[:max_len]
    out = np.zeros(max_len, dtype=np.float32)
    out[:len(x)] = x
    return out

class TrialDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = pad_or_truncate(row["pupil"], self.max_len)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, T)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return {
            "x": x,
            "y": y,
            "subject_id": int(row["subject_id"]),
            "trial": int(row["trial"]),
        }