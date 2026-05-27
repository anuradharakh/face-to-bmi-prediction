import sys
import json
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append("src")

from face_bmi.config import load_config
from face_bmi.data.dataset import FaceBMIDataset
from face_bmi.models.m3_efficientnet_finetune import EfficientNetBMIRegressor
from face_bmi.training.metrics import regression_metrics
from face_bmi.utils import get_device


def plot_training_curve(history_path, output_path):
    history = json.loads(Path(history_path).read_text())

    epochs = [h["epoch"] for h in history]
    train_mae = [h["train"]["mae"] for h in history]
    test_mae = [h["test"]["mae"] for h in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_mae, marker="o", label="Train MAE")
    plt.plot(epochs, test_mae, marker="o", label="Validation/Test MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("EfficientNet-B0 Training vs Validation MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predicted_vs_actual(cfg, output_path):
    device = get_device(cfg["training"]["device"])

    dataset = FaceBMIDataset(
        csv_path="data/processed/test.csv",
        image_size=cfg["data"]["image_size"],
        is_train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    m3_cfg = cfg["models"]["m3"]

    model = EfficientNetBMIRegressor(
        hidden_units=tuple(m3_cfg["head_hidden_units"]),
        dropout=m3_cfg["dropout"],
        unfreeze_last_n_blocks=m3_cfg["unfreeze_last_n_blocks"],
        backbone_name=m3_cfg.get("backbone", "efficientnet_b0"),
    ).to(device)

    model.load_state_dict(
        torch.load("models/efficientnet_finetuned.pt", map_location=device)
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            bmi = batch["bmi"].to(device)

            preds = model(images)

            y_true.extend(bmi.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = regression_metrics(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        linestyle="--",
        label="Ideal prediction",
    )
    plt.xlabel("Actual BMI")
    plt.ylabel("Predicted BMI")
    plt.title(f"EfficientNet-B0 Predicted vs Actual BMI\nPearson r = {metrics['pearson_r']:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(metrics)


def main():
    cfg = load_config()

    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    plot_training_curve(
        history_path="outputs/metrics/efficientnet_finetune_history.json",
        output_path="outputs/plots/efficientnet_mae_curve.png",
    )

    plot_predicted_vs_actual(
        cfg=cfg,
        output_path="outputs/plots/efficientnet_predicted_vs_actual.png",
    )

    print("Saved plots:")
    print(" - outputs/plots/efficientnet_mae_curve.png")
    print(" - outputs/plots/efficientnet_predicted_vs_actual.png")


if __name__ == "__main__":
    main()