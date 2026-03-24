# train_eval.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

from DengPaper.build_trial_examples import build_examples
from dataset_utils import TrialDataset
from model import PupilCNN

def compute_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "auc": auc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
    }

def aggregate_subject_level(subject_ids, probs, labels):
    df = pd.DataFrame({
        "subject_id": subject_ids,
        "prob": probs,
        "label": labels,
    })
    agg = df.groupby("subject_id", as_index=False).agg({
        "prob": "mean",
        "label": "first",
    })
    return agg["label"].to_numpy(), agg["prob"].to_numpy()

def train_one_fold(train_df, test_df, device="cuda", epochs=15, batch_size=64, lr=1e-3):
    max_len = int(np.quantile([len(x) for x in train_df["pupil"]], 0.95))
    max_len = max(128, min(max_len, 4000))

    train_ds = TrialDataset(train_df, max_len=max_len)
    test_ds = TrialDataset(test_df, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = PupilCNN().to(device)

    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_probs, all_labels, all_subjects = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(batch["y"].numpy().astype(int).tolist())
            all_subjects.extend(batch["subject_id"].numpy().tolist())

    trial_metrics = compute_metrics(all_labels, all_probs)
    subj_y, subj_p = aggregate_subject_level(all_subjects, all_probs, all_labels)
    subject_metrics = compute_metrics(subj_y, subj_p)

    return trial_metrics, subject_metrics

def main():
    mat_path = "Pupil_dataset.mat"
    df = build_examples(mat_path)

    # Binary ADHD vs Control
    groups = df["subject_id"].to_numpy()
    y = df["label"].to_numpy()

    logo = LeaveOneGroupOut()
    trial_results = []
    subject_results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for fold, (tr_idx, te_idx) in enumerate(logo.split(df, y, groups), start=1):
        train_df = df.iloc[tr_idx].copy()
        test_df = df.iloc[te_idx].copy()

        # skip if train fold has only one class
        if train_df["label"].nunique() < 2:
            continue

        trial_metrics, subject_metrics = train_one_fold(train_df, test_df, device=device)
        print(f"Fold {fold}")
        print(" trial:", trial_metrics)
        print(" subject:", subject_metrics)

        trial_results.append(trial_metrics)
        subject_results.append(subject_metrics)

    print("\nSubject-level average:")
    for key in ["auc", "balanced_accuracy", "f1", "sensitivity", "specificity"]:
        vals = [r[key] for r in subject_results if not np.isnan(r[key])]
        print(key, np.mean(vals), np.std(vals))

if __name__ == "__main__":
    main()