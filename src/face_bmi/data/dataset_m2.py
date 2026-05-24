"""
M2 Dataset & Augmentation (Improved)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import logging
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader
import torchvision.transforms.v2 as T

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────
BMI_BINS = [0, 18.5, 25.0, 30.0, 35.0, 100.0]
BMI_MEAN = 27.0
BMI_STD  = 6.0

# All known string representations → int
GENDER_MAP = {
    "male": 1, "m": 1, "1": 1,
    "female": 0, "f": 0, "0": 0,
}


# ─── Augmentation pipelines ───────────────────────────────────────────────────
def build_train_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size + 16, img_size + 16)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([
            T.RandomAffine(
                degrees=8,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=4,
            )
        ], p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.2),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.5, 2.0)),
    ])


def build_val_transforms(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def _parse_gender(value) -> int:
    """Robustly convert any gender representation to 0/1 int."""
    if isinstance(value, (int, float)):
        return int(value)
    return GENDER_MAP.get(str(value).strip().lower(), 0)


# ─── Dataset ─────────────────────────────────────────────────────────────────
class BMIFaceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        split: str = "train",
        transform=None,
        bmi_mean: float = BMI_MEAN,
        bmi_std:  float = BMI_STD,
        label_noise_std: float = 0.0,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.bmi_mean = bmi_mean
        self.bmi_std  = bmi_std
        self.label_noise_std = label_noise_std

        df = pd.read_csv(csv_path)

        # FIX: normalise gender to int once, here in __init__
        df["gender"] = df["gender"].apply(_parse_gender)

        is_train_flag = df["is_training"].astype(bool)
        split_df = (
            df[is_train_flag].reset_index(drop=True) if split == "train"
            else df[~is_train_flag].reset_index(drop=True)
        )

        # ── Drop rows whose image file does not exist on disk ────────────────
        exists_mask = split_df["name"].apply(
            lambda name: (self.img_dir / name).exists()
        )
        n_missing = (~exists_mask).sum()
        if n_missing > 0:
            missing_names = split_df.loc[~exists_mask, "name"].tolist()
            logger.warning(
                f"[BMIFaceDataset/{split}] Dropping {n_missing} rows with "
                f"missing image files. First 5: {missing_names[:5]}"
            )
        self.df = split_df[exists_mask].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError(
                f"[BMIFaceDataset/{split}] No valid images found in {self.img_dir}. "
                f"Check that img_dir points to the correct folder."
            )

        self.bmi_bins = pd.cut(
            self.df["bmi"],
            bins=BMI_BINS,
            labels=False,
            right=False,
        ).fillna(0).astype(int).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["name"]

        # Safe image load — returns black image on corruption instead of crashing
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}. Using blank fallback.")
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)

        bmi_raw = float(row["bmi"])
        if self.label_noise_std > 0:
            bmi_raw += random.gauss(0, self.label_noise_std)

        bmi_norm = (bmi_raw - self.bmi_mean) / self.bmi_std

        # FIX: gender is already int from __init__ — no int() conversion needed
        gender = row["gender"]

        return (
            img,
            torch.tensor(bmi_norm, dtype=torch.float32),
            torch.tensor(gender,   dtype=torch.float32),
        )


# ─── Stratified sampler ───────────────────────────────────────────────────────
class StratifiedBMISampler(Sampler):
    def __init__(self, bmi_bins: list[int], batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        bins_arr = np.array(bmi_bins)
        self.class_indices = {
            c: np.where(bins_arr == c)[0].tolist()
            for c in np.unique(bins_arr)
        }
        self.n_samples = len(bmi_bins)

    def __iter__(self):
        per_class = {c: random.sample(idxs, len(idxs))
                     for c, idxs in self.class_indices.items()}
        combined, pointers = [], {c: 0 for c in per_class}
        while True:
            added = 0
            for c, idxs in per_class.items():
                p = pointers[c]
                if p < len(idxs):
                    combined.append(idxs[p])
                    pointers[c] += 1
                    added += 1
            if added == 0:
                break
        return iter(combined)

    def __len__(self):
        return self.n_samples


# ─── MixUp ───────────────────────────────────────────────────────────────────
def mixup_batch(imgs, bmi, gender, alpha: float = 0.4):
    lam  = np.random.beta(alpha, alpha)
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    return (
        lam * imgs   + (1 - lam) * imgs[perm],
        lam * bmi    + (1 - lam) * bmi[perm],
        gender if lam >= 0.5 else gender[perm],
    )


# ─── Factory helpers ──────────────────────────────────────────────────────────
def compute_bmi_stats(csv_path: str) -> tuple[float, float]:
    df = pd.read_csv(csv_path)
    train_bmi = df.loc[df["is_training"].astype(bool), "bmi"]
    return float(train_bmi.mean()), float(train_bmi.std())


def check_missing_files(csv_path: str, img_dir: str) -> None:
    """Print a summary of missing image files before training starts."""
    df = pd.read_csv(csv_path)
    img_dir = Path(img_dir)
    missing = df["name"].apply(lambda n: not (img_dir / n).exists())
    n_missing = missing.sum()
    if n_missing == 0:
        print(f"[DataCheck] All {len(df)} image files found.")
    else:
        print(f"[DataCheck] WARNING: {n_missing}/{len(df)} image files missing.")
        print(f"            First 5 missing: {df.loc[missing, 'name'].tolist()[:5]}")
        print(f"            These rows will be skipped during training.")


def _is_mps() -> bool:
    return torch.backends.mps.is_available()


def make_dataloaders(
    csv_path: str,
    img_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    label_noise_std: float = 0.5,
    use_stratified_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, float, float]:
    bmi_mean, bmi_std = compute_bmi_stats(csv_path)
    check_missing_files(csv_path, img_dir)

    train_ds = BMIFaceDataset(
        csv_path, img_dir, split="train",
        transform=build_train_transforms(img_size),
        bmi_mean=bmi_mean, bmi_std=bmi_std,
        label_noise_std=label_noise_std,
    )
    val_ds = BMIFaceDataset(
        csv_path, img_dir, split="val",
        transform=build_val_transforms(img_size),
        bmi_mean=bmi_mean, bmi_std=bmi_std,
        label_noise_std=0.0,
    )

    # FIX: pin_memory=False on MPS (not supported, causes warning)
    pin = not _is_mps()

    if use_stratified_sampler:
        sampler = StratifiedBMISampler(train_ds.bmi_bins, batch_size)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, bmi_mean, bmi_std
