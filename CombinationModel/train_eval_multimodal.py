import argparse
import copy
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from models import (
    ChoiPupilNet,
    DengGazeNet,
    FusionADHDNet,
    choi_loss,
    deng_loss,
    fusion_loss,
    prepare_masked_fusion_batch,
)
from multimodal_dataloaders import (
    build_all_loaders_for_fair_comparison,
    build_choi_only_loaders,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def make_pos_weight_from_loader(loader, device: torch.device) -> torch.Tensor | None:
    labels = []
    for batch in loader:
        y = batch["label"]
        labels.append(y.detach().cpu().numpy())
    if not labels:
        return None
    labels = np.concatenate(labels).astype(np.float64)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos <= 0 or neg <= 0:
        return None
    return torch.tensor(neg / pos, dtype=torch.float32, device=device)


@torch.no_grad()
def collect_predictions(
    model_name: str,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    apply_masked_fusion_eval: bool = False,
    fusion_recon_weight: float = 0.15,
    pos_weight: torch.Tensor | None = None,
) -> Dict[str, np.ndarray]:
    model.eval()

    probs_all: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    subjects_all: List[np.ndarray] = []
    trials_all: List[np.ndarray] = []
    losses: List[float] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if model_name == "choi":
            outputs = model(batch["pupil"], batch["pupil_obs_mask"])
            loss_dict = choi_loss(outputs, batch, pos_weight=pos_weight)

        elif model_name == "deng":
            outputs = model(batch["gaze"], batch["gaze_obs_mask"])
            loss_dict = deng_loss(outputs, batch, pos_weight=pos_weight)

        elif model_name == "fusion":
            if apply_masked_fusion_eval:
                eval_batch = prepare_masked_fusion_batch(batch, keep_prob=0.90)
            else:
                eval_batch = dict(batch)
                eval_batch["pupil_recon_target"] = batch["pupil"]
                eval_batch["pupil_recon_target_mask"] = batch["pupil_obs_mask"]

            outputs = model(
                eval_batch["pupil"],
                eval_batch["pupil_obs_mask"],
                eval_batch["gaze"],
                eval_batch["gaze_obs_mask"],
            )
            loss_dict = fusion_loss(outputs, eval_batch, recon_weight=fusion_recon_weight, pos_weight=pos_weight)

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        probs = torch.sigmoid(outputs["logits"]).detach().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)

        labels = batch["label"].detach().cpu().numpy()
        subjects = batch["subject_id"].detach().cpu().numpy()
        trials = batch["trial"].detach().cpu().numpy()

        probs_all.append(probs)
        labels_all.append(labels)
        subjects_all.append(subjects)
        trials_all.append(trials)
        losses.append(float(loss_dict["loss"].detach().cpu().item()))

    return {
        "probs": np.concatenate(probs_all),
        "labels": np.concatenate(labels_all),
        "subject_id": np.concatenate(subjects_all),
        "trial": np.concatenate(trials_all),
        "loss": float(np.mean(losses)) if losses else np.nan,
    }


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    y_pred = (y_prob >= threshold).astype(int)

    specificity = np.nan
    neg_mask = y_true == 0
    if neg_mask.sum() > 0:
        specificity = float(((y_pred[neg_mask] == 0).sum()) / neg_mask.sum())

    sensitivity = np.nan
    pos_mask = y_true == 1
    if pos_mask.sum() > 0:
        sensitivity = float(((y_pred[pos_mask] == 1).sum()) / pos_mask.sum())

    out = {
        "auc": safe_auc(y_true, y_prob),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
    return out


def aggregate_subjectwise(preds: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    subjects = preds["subject_id"]
    probs = preds["probs"]
    labels = preds["labels"]

    uniq = np.unique(subjects)
    subject_probs = []
    subject_labels = []
    subject_ids = []

    for sid in uniq:
        idx = subjects == sid
        subject_ids.append(sid)
        subject_probs.append(float(np.mean(probs[idx])))
        labs = labels[idx]
        subject_labels.append(int(round(float(np.mean(labs)))))

    return np.asarray(subject_ids), np.asarray(subject_labels), np.asarray(subject_probs)


def tune_threshold_from_predictions(preds: Dict[str, np.ndarray]) -> float:
    _, subj_labels, subj_probs = aggregate_subjectwise(preds)
    candidates = np.linspace(0.2, 0.8, 61)
    best_thr = 0.5
    best_score = -np.inf
    for thr in candidates:
        metrics = compute_binary_metrics(subj_labels, subj_probs, threshold=float(thr))
        score = metrics["balanced_accuracy"]
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def evaluate_predictions(preds: Dict[str, np.ndarray], threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    trial_metrics = compute_binary_metrics(preds["labels"], preds["probs"], threshold=threshold)

    _, subj_labels, subj_probs = aggregate_subjectwise(preds)
    subject_metrics = compute_binary_metrics(subj_labels, subj_probs, threshold=threshold)

    return {
        "trial": trial_metrics,
        "subject": subject_metrics,
    }


def train_one_epoch(
    model_name: str,
    model: torch.nn.Module,
    loader,
    optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None = None,
    fusion_keep_prob: float = 0.90,
    fusion_recon_weight: float = 0.15,
) -> float:
    model.train()
    losses = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if model_name == "choi":
            outputs = model(batch["pupil"], batch["pupil_obs_mask"])
            loss_dict = choi_loss(outputs, batch, pos_weight=pos_weight)

        elif model_name == "deng":
            outputs = model(batch["gaze"], batch["gaze_obs_mask"])
            loss_dict = deng_loss(outputs, batch, pos_weight=pos_weight)

        elif model_name == "fusion":
            masked_batch = prepare_masked_fusion_batch(batch, keep_prob=fusion_keep_prob)
            outputs = model(
                masked_batch["pupil"],
                masked_batch["pupil_obs_mask"],
                masked_batch["gaze"],
                masked_batch["gaze_obs_mask"],
            )
            loss_dict = fusion_loss(outputs, masked_batch, recon_weight=fusion_recon_weight, pos_weight=pos_weight)

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        loss = loss_dict["loss"]

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else np.nan


def make_model(model_name: str) -> torch.nn.Module:
    if model_name == "choi":
        return ChoiPupilNet(
            hidden_channels=(64, 128, 128),
            kernel_size=5,
            dropout=0.1,
            classifier_hidden=128,
        )
    if model_name == "deng":
        return DengGazeNet(
            in_channels=4,
            hidden_channels=(64, 128, 128),
            kernel_size=5,
            dropout=0.1,
            classifier_hidden=128,
        )
    if model_name == "fusion":
        return FusionADHDNet(
            pupil_hidden=(64, 128, 128),
            gaze_hidden=(64, 128, 128),
            kernel_size=5,
            dropout=0.15,
            gaze_in_channels=4,
            d_model=64,
            num_heads=2,
            num_cross_blocks=1,
            fusion_len=192,
            classifier_hidden=128,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def get_loaders(args):
    if args.model == "choi" and args.choi_all_pupil_rows:
        bundle = build_choi_only_loaders(
            pkl_path=args.data,
            batch_size=args.batch_size,
            pupil_len=args.pupil_len,
            seed=args.seed,
            num_workers=args.num_workers,
            use_all_pupil_rows=True,
        )
        return bundle["choi"]

    bundle = build_all_loaders_for_fair_comparison(
        pkl_path=args.data,
        batch_size=args.batch_size,
        pupil_len=args.pupil_len,
        gaze_len=args.gaze_len,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    return bundle[args.model]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["choi", "deng", "fusion"], required=True)
    parser.add_argument("--data", type=str, default="multimodal_trials.pkl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pupil_len", type=int, default=512)
    parser.add_argument("--gaze_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--choi_all_pupil_rows", action="store_true")
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--fusion_keep_prob", type=float, default=0.90)
    parser.add_argument("--fusion_recon_weight", type=float, default=0.15)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.lr is None:
        args.lr = 3e-4 if args.model == "fusion" else 1e-3

    loaders = get_loaders(args)
    model = make_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    pos_weight = make_pos_weight_from_loader(loaders["train"], device) if args.use_pos_weight else None

    best_state = None
    best_val_auc = -float("inf")
    best_epoch = -1
    best_threshold = 0.5
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            args.model,
            model,
            loaders["train"],
            optimizer,
            device,
            pos_weight=pos_weight,
            fusion_keep_prob=args.fusion_keep_prob,
            fusion_recon_weight=args.fusion_recon_weight,
        )

        train_preds = collect_predictions(
            args.model,
            model,
            loaders["train"],
            device,
            apply_masked_fusion_eval=False,
            fusion_recon_weight=args.fusion_recon_weight,
            pos_weight=pos_weight,
        )
        val_preds = collect_predictions(
            args.model,
            model,
            loaders["val"],
            device,
            apply_masked_fusion_eval=False,
            fusion_recon_weight=args.fusion_recon_weight,
            pos_weight=pos_weight,
        )

        tuned_threshold = tune_threshold_from_predictions(val_preds)
        train_metrics = evaluate_predictions(train_preds, threshold=tuned_threshold)
        val_metrics = evaluate_predictions(val_preds, threshold=tuned_threshold)

        val_auc = val_metrics["subject"]["auc"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_preds["loss"],
            "train_subject_auc": train_metrics["subject"]["auc"],
            "val_subject_auc": val_auc,
            "train_subject_balanced_accuracy": train_metrics["subject"]["balanced_accuracy"],
            "val_subject_balanced_accuracy": val_metrics["subject"]["balanced_accuracy"],
            "train_trial_auc": train_metrics["trial"]["auc"],
            "val_trial_auc": val_metrics["trial"]["auc"],
            "threshold": tuned_threshold,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_subj_auc={val_metrics['subject']['auc']:.4f} | "
            f"val_subj_bacc={val_metrics['subject']['balanced_accuracy']:.4f} | "
            f"val_trial_auc={val_metrics['trial']['auc']:.4f} | "
            f"thr={tuned_threshold:.2f}"
        )

        improved = np.isfinite(val_auc) and (val_auc > best_val_auc)
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            best_threshold = tuned_threshold
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(val_auc if np.isfinite(val_auc) else -1e9)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_preds = collect_predictions(
        args.model,
        model,
        loaders["test"],
        device,
        apply_masked_fusion_eval=False,
        fusion_recon_weight=args.fusion_recon_weight,
        pos_weight=pos_weight,
    )
    test_metrics = evaluate_predictions(test_preds, threshold=best_threshold)

    summary = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_subject_auc": best_val_auc,
        "best_threshold": best_threshold,
        "test_trial_metrics": test_metrics["trial"],
        "test_subject_metrics": test_metrics["subject"],
        "n_test_trials": int(len(test_preds["labels"])),
        "n_test_subjects": int(len(np.unique(test_preds["subject_id"]))),
        "args": vars(args),
    }

    stem = f"{args.model}_seed{args.seed}"
    ckpt_path = os.path.join(args.outdir, f"{stem}.pt")
    hist_path = os.path.join(args.outdir, f"{stem}_history.json")
    summary_path = os.path.join(args.outdir, f"{stem}_summary.json")

    torch.save(model.state_dict(), ckpt_path)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBest checkpoint saved to:", ckpt_path)
    print("History saved to:", hist_path)
    print("Summary saved to:", summary_path)
    print("Best threshold:", best_threshold)
    print("\nTest metrics (trial-level):")
    print(json.dumps(test_metrics["trial"], indent=2))
    print("\nTest metrics (subject-level):")
    print(json.dumps(test_metrics["subject"], indent=2))


if __name__ == "__main__":
    main()