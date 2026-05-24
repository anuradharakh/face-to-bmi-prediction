#!/usr/bin/env python3
"""
scripts/04_eval_m2.py
=====================
Loads the best M2 checkpoint and evaluates on the val split.
Reports overall R², RMSE, MAE, and a per-BMI-category breakdown.
"""

import sys
import argparse
import yaml
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_bmi.models.model_m2   import ResNet50MultiTask, MultiTaskLoss
from src.face_bmi.data.dataset_m2 import BMIFaceDataset, build_val_transforms, BMI_BINS
from src.face_bmi.training.trainer_m2 import get_device
from torch.utils.data import DataLoader


def evaluate(cfg: dict, checkpoint: str):
    device = get_device()

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    bmi_mean = ckpt["bmi_mean"]
    bmi_std  = ckpt["bmi_std"]

    model = ResNet50MultiTask().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ds = BMIFaceDataset(
        cfg["csv_path"], cfg["img_dir"], split="val",
        transform=build_val_transforms(cfg.get("img_size", 224)),
        bmi_mean=bmi_mean, bmi_std=bmi_std,
    )
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    all_pred, all_true, all_bins = [], [], []

    with torch.no_grad():
        for imgs, bmi_norm, gender in loader:
            imgs = imgs.to(device)
            bmi_pred_norm, _ = model(imgs)
            # Inverse z-score to get raw BMI
            pred_raw = bmi_pred_norm.cpu().numpy() * bmi_std + bmi_mean
            true_raw = bmi_norm.numpy() * bmi_std + bmi_mean
            all_pred.extend(pred_raw)
            all_true.extend(true_raw)

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    bins = pd.cut(all_true, bins=BMI_BINS, labels=["Underweight","Normal","Overweight","Obese I","Obese II+"], right=False)

    # ── Overall metrics ──────────────────────────────────────────────────────
    r2   = r2_score(all_true, all_pred)
    rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    mae  = mean_absolute_error(all_true, all_pred)

    print(f"\n{'='*50}")
    print(f"  M2 Evaluation — {Path(checkpoint).name}")
    print(f"{'='*50}")
    print(f"  Overall R²  : {r2:.4f}")
    print(f"  RMSE        : {rmse:.3f} BMI units")
    print(f"  MAE         : {mae:.3f} BMI units")
    print(f"{'='*50}")

    # ── Per-category breakdown ───────────────────────────────────────────────
    print("\n  Per-BMI-category breakdown:")
    print(f"  {'Category':<18} {'N':>5} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print(f"  {'-'*50}")
    for cat in ["Underweight","Normal","Overweight","Obese I","Obese II+"]:
        mask = (bins == cat).values
        if mask.sum() < 5:
            print(f"  {cat:<18} {mask.sum():>5}    (too few samples)")
            continue
        cat_r2   = r2_score(all_true[mask], all_pred[mask])
        cat_rmse = np.sqrt(mean_squared_error(all_true[mask], all_pred[mask]))
        cat_mae  = mean_absolute_error(all_true[mask], all_pred[mask])
        print(f"  {cat:<18} {mask.sum():>5} {cat_r2:>8.4f} {cat_rmse:>8.3f} {cat_mae:>8.3f}")

    return r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/m2_resnet50.yaml")
    parser.add_argument("--checkpoint", default="reports/m2/m2_best.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.checkpoint)
