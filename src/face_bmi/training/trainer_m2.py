"""
M2 Trainer (Improved)
======================
Training loop with:
  1. Two-phase schedule: freeze backbone → unfreeze layer4 at epoch N
  2. Cosine annealing LR  (separately for backbone vs head params)
  3. Gradient clipping  (prevents exploding grads after unfreeze)
  4. MixUp every-other-batch  (p=0.5)
  5. R² tracked live; best model saved on val R²
  6. Apple MPS / CUDA / CPU autodetect
  7. W&B / CSV logging  (W&B optional; always logs to CSV)
"""

from __future__ import annotations

import math
import time
import csv
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from sklearn.metrics import r2_score
import numpy as np

from src.face_bmi.models.model_m2  import ResNet50MultiTask, MultiTaskLoss
from src.face_bmi.data.dataset_m2 import make_dataloaders, mixup_batch


# ─── Device helper ────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Parameter groups ────────────────────────────────────────────────────────
def make_param_groups(model: ResNet50MultiTask, cfg: dict) -> list[dict]:
    """
    Three groups:
      backbone  → tiny LR (fine-tune carefully)
      heads     → normal LR
      task weights (log_sigma) → separate LR
    """
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    loss_ids     = {id(p) for p in [model.loss_fn.log_sigma_bmi,   # attached below
                                     model.loss_fn.log_sigma_gender]}

    backbone_params  = [p for p in model.parameters() if id(p) in backbone_ids]
    head_params      = [p for p in model.parameters()
                        if id(p) not in backbone_ids and id(p) not in loss_ids]

    return [
        {"params": backbone_params, "lr": cfg["lr_backbone"],  "name": "backbone"},
        {"params": head_params,     "lr": cfg["lr_head"],      "name": "heads"},
    ]


# ─── Single epoch ─────────────────────────────────────────────────────────────
def run_epoch(
    model:    ResNet50MultiTask,
    loader,
    loss_fn:  MultiTaskLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device:   torch.device,
    is_train: bool,
    mixup_prob: float = 0.5,
    clip_grad: float = 1.0,
) -> dict:
    model.train() if is_train else model.eval()

    total_loss = bmi_loss_sum = gender_loss_sum = 0.0
    all_bmi_pred, all_bmi_true = [], []
    all_gender_logit, all_gender_true = [], []
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, bmi_norm, gender in loader:
            imgs       = imgs.to(device, non_blocking=True)
            bmi_norm   = bmi_norm.to(device, non_blocking=True)
            gender     = gender.to(device, non_blocking=True)

            # MixUp (train only, 50% of batches)
            if is_train and torch.rand(1).item() < mixup_prob:
                imgs, bmi_norm, gender = mixup_batch(imgs, bmi_norm, gender)

            bmi_pred, gender_logit = model(imgs)

            loss, l_bmi, l_gender = loss_fn(bmi_pred, bmi_norm, gender_logit, gender)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            total_loss      += loss.item()
            bmi_loss_sum    += l_bmi.item()
            gender_loss_sum += l_gender.item()

            all_bmi_pred.extend(bmi_pred.detach().cpu().numpy())
            all_bmi_true.extend(bmi_norm.detach().cpu().numpy())
            all_gender_logit.extend(gender_logit.detach().cpu().numpy())
            all_gender_true.extend(gender.detach().cpu().numpy())
            n_batches += 1

    # R² on normalised BMI (same as on raw BMI — z-score doesn't change R²)
    r2 = r2_score(all_bmi_true, all_bmi_pred)

    gender_acc = (
        (np.array(all_gender_logit) > 0).astype(int) ==
        np.array(all_gender_true).astype(int)
    ).mean()

    return {
        "loss":       total_loss / n_batches,
        "bmi_loss":   bmi_loss_sum / n_batches,
        "gender_loss":gender_loss_sum / n_batches,
        "r2":         r2,
        "gender_acc": gender_acc,
    }


# ─── Main train function ──────────────────────────────────────────────────────
def train_m2(cfg: dict, save_dir: str = "reports/m2"):
    """
    cfg keys (all available in configs/m2_resnet50.yaml):
      csv_path, img_dir, batch_size, img_size, num_workers,
      epochs_phase1, epochs_phase2, epochs_phase3 (optional),
      lr_backbone, lr_head, weight_decay,
      label_noise_std, mixup_prob, clip_grad,
      use_uncertainty_weighting, huber_delta
    """
    device = get_device()
    print(f"[M2] Using device: {device}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, bmi_mean, bmi_std = make_dataloaders(
        csv_path    = cfg["csv_path"],
        img_dir     = cfg["img_dir"],
        batch_size  = cfg["batch_size"],
        img_size    = cfg.get("img_size", 224),
        num_workers = cfg.get("num_workers", 4),
        label_noise_std = cfg.get("label_noise_std", 0.5),
        use_stratified_sampler = cfg.get("use_stratified_sampler", True),
    )
    print(f"[M2] BMI stats — mean: {bmi_mean:.2f}, std: {bmi_std:.2f}")

    # ── Model + Loss ─────────────────────────────────────────────────────────
    model = ResNet50MultiTask(
        dropout_shared  = cfg.get("dropout_shared",  0.3),
        dropout_bmi     = cfg.get("dropout_bmi",     0.4),
        dropout_gender  = cfg.get("dropout_gender",  0.3),
    ).to(device)

    loss_fn = MultiTaskLoss(
        use_uncertainty_weighting = cfg.get("use_uncertainty_weighting", True),
        huber_delta               = cfg.get("huber_delta", 5.0),
    ).to(device)

    # Attach loss_fn to model so param group builder can see it
    model.loss_fn = loss_fn

    # ── Phase 1: Freeze backbone, train heads ────────────────────────────────
    model.freeze_backbone()
    phase1_epochs = cfg.get("epochs_phase1", 10)
    phase2_epochs = cfg.get("epochs_phase2", 30)
    phase3_epochs = cfg.get("epochs_phase3", 0)
    total_epochs  = phase1_epochs + phase2_epochs + phase3_epochs

    param_groups = make_param_groups(model, cfg)
    # Also include loss_fn parameters
    param_groups.append({
        "params": list(loss_fn.parameters()),
        "lr": cfg["lr_head"],
        "name": "task_weights",
    })

    optimizer = AdamW(param_groups, weight_decay=cfg.get("weight_decay", 1e-4))

    # Warmup for 2 epochs then cosine decay
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=2)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - 2, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])

    best_r2   = -9999.0
    log_rows  = []

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        # ── Phase transitions ────────────────────────────────────────────
        if epoch == phase1_epochs + 1:
            print(f"\n[M2] Epoch {epoch}: Unfreezing layer4, reducing backbone LR")
            model.unfreeze_layer4()
            # Set backbone group to fine-tune LR (very small)
            for pg in optimizer.param_groups:
                if pg.get("name") == "backbone":
                    pg["lr"] = cfg["lr_backbone"]

        if phase3_epochs > 0 and epoch == phase1_epochs + phase2_epochs + 1:
            print(f"\n[M2] Epoch {epoch}: Unfreezing layer3 + layer4")
            model.unfreeze_layer3_and_4()

        # ── Train & validate ─────────────────────────────────────────────
        train_metrics = run_epoch(
            model, train_loader, loss_fn, optimizer, device,
            is_train=True,
            mixup_prob=cfg.get("mixup_prob", 0.5),
            clip_grad=cfg.get("clip_grad", 1.0),
        )
        val_metrics = run_epoch(
            model, val_loader, loss_fn, None, device,
            is_train=False,
        )

        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[1]["lr"]   # head LR

        print(
            f"Epoch {epoch:03d}/{total_epochs} | "
            f"train R²={train_metrics['r2']:.4f} loss={train_metrics['loss']:.4f} | "
            f"val R²={val_metrics['r2']:.4f} loss={val_metrics['loss']:.4f} | "
            f"gender acc={val_metrics['gender_acc']:.3f} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        # ── Save best ────────────────────────────────────────────────────
        if val_metrics["r2"] > best_r2:
            best_r2 = val_metrics["r2"]
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "loss_fn":  loss_fn.state_dict(),
                "optimizer":optimizer.state_dict(),
                "bmi_mean": bmi_mean,
                "bmi_std":  bmi_std,
                "val_r2":   best_r2,
            }, save_path / "m2_best.pt")
            print(f"  ★ New best val R²: {best_r2:.4f}")

        log_rows.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        })

    # ── Save log CSV ─────────────────────────────────────────────────────────
    if log_rows:
        keys = log_rows[0].keys()
        with open(save_path / "m2_training_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"\n[M2] Training complete. Best val R²: {best_r2:.4f}")
    print(f"[M2] Model saved to {save_path / 'm2_best.pt'}")
    return best_r2
