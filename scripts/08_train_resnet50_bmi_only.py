import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("src")

from face_bmi.config import load_config
from face_bmi.utils import set_seed, get_device
from face_bmi.data.dataset import FaceBMIDataset
from face_bmi.models.m4_resnet50_bmi_only import ResNet50BMIOnly
from face_bmi.training.metrics import regression_metrics
from face_bmi.training.onnx_export import (
    export_bmi_model_to_onnx,
    export_multitask_model_to_onnx,
)

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        bmi = batch["bmi"].to(device)

        with torch.set_grad_enabled(train):
            preds = model(images)
            loss = criterion(preds, bmi)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        y_true.extend(bmi.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    metrics = regression_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    train_cfg = cfg["training"]["resnet50_bmi_only"]

    batch_size = train_cfg["batch_size"]
    epochs = train_cfg["epochs"]
    learning_rate = train_cfg["learning_rate"]
    weight_decay = train_cfg["weight_decay"]
    num_workers = cfg["training"]["num_workers"]

    device = get_device(cfg["training"]["device"])
    print(f"Using device: {device}")

    train_dataset = FaceBMIDataset(
        csv_path="data/processed/train.csv",
        image_size=cfg["data"]["image_size"],
        is_train=True,
    )

    test_dataset = FaceBMIDataset(
        csv_path="data/processed/test.csv",
        image_size=cfg["data"]["image_size"],
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model_cfg = cfg["models"]["resnet50_bmi_only"]

    model = ResNet50BMIOnly(
        hidden_units=tuple(model_cfg["head_hidden_units"]),
        dropout=model_cfg["dropout"],
        frozen=model_cfg["frozen"],
        fine_tune_last_block=model_cfg["fine_tune_last_block"],
    ).to(device)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_pearson = -999
    history = []

    Path("models").mkdir(exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "test": test_metrics,
            }
        )

        print(
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Test MAE: {test_metrics['mae']:.4f} | "
            f"Test RMSE: {test_metrics['rmse']:.4f} | "
            f"Test Pearson r: {test_metrics['pearson_r']:.4f} | "
            f"Test R2: {test_metrics['r2']:.4f}"
        )

        if test_metrics["pearson_r"] > best_pearson:
            best_pearson = test_metrics["pearson_r"]

            pt_path = "models/resnet50_bmi_only_best.pt"
            onnx_path = "models/resnet50_bmi_only_best.onnx"

            torch.save(model.state_dict(), pt_path)
            export_bmi_model_to_onnx(model, onnx_path, device)
            model.train()

            print("Saved new best ResNet50 BMI-only model.")
            print(f"Saved PT: {pt_path}")
            print(f"Saved ONNX: {onnx_path}")

    with open(
        "outputs/metrics/resnet50_bmi_only_history.json",
        "w",
    ) as f:
        json.dump(history, f, indent=2)

    print("\nResNet50 BMI-only training complete.")
    print(f"Best Pearson r: {best_pearson:.4f}")


if __name__ == "__main__":
    main()