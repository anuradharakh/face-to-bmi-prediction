#!/usr/bin/env python3
"""
scripts/03_train_m2.py
======================
Run:  python scripts/03_train_m2.py
      python scripts/03_train_m2.py --config configs/config.yaml

Loads the M2 section from the shared config.yaml and runs the improved
M2 training pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path (works when run from repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_bmi.config import load_config
from src.face_bmi.training.trainer_m2 import train_m2


def main():
    parser = argparse.ArgumentParser(description="Train M2 ResNet50 multi-task model")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to shared config.yaml (uses the 'm2' section)",
    )
    # Optional CLI overrides — take precedence over config.yaml values
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--epochs_phase1", type=int,   default=None)
    parser.add_argument("--epochs_phase2", type=int,   default=None)
    parser.add_argument("--epochs_phase3", type=int,   default=None)
    parser.add_argument("--lr_head",       type=float, default=None)
    parser.add_argument("--lr_backbone",   type=float, default=None)
    parser.add_argument("--mixup_prob",    type=float, default=None)
    parser.add_argument("--no_stratify",   action="store_true",
                        help="Disable stratified BMI sampler")
    args = parser.parse_args()

    # Load full config then extract the m2 section
    full_cfg = load_config(args.config)
    cfg = full_cfg["m2"]

    # Apply any CLI overrides
    for key in ["batch_size", "epochs_phase1", "epochs_phase2",
                "epochs_phase3", "lr_head", "lr_backbone", "mixup_prob"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    if args.no_stratify:
        cfg["use_stratified_sampler"] = False

    print("=" * 60)
    print("M2 ResNet50 Multi-Task — Improved Training")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    final_r2 = train_m2(cfg, save_dir=cfg.get("save_dir", "reports/m2"))

    if final_r2 >= 0.68:
        print(f"\n✓ Target achieved: val R² = {final_r2:.4f} ≥ 0.68")
    else:
        print(f"\n✗ Val R² = {final_r2:.4f} — below 0.68 target")
        print("  Try: increasing epochs_phase2, enabling phase3, or reducing label_noise_std")


if __name__ == "__main__":
    main()