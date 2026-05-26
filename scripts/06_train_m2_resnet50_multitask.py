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
from face_bmi.models.m2_resnet50_multitask import ResNet50MultiTaskBMI
from face_bmi.training.metrics import regression_metrics
from face_bmi.training.onnx_export import (
    export_bmi_model_to_onnx,
    export_multitask_model_to_onnx,
)


def run_epoch(
    model,
    loader,
    bmi_criterion,
    gender_criterion,
    optimizer,
    device,
    bmi_loss_weight=1.0,
    gender_loss_weight=0.2,
    train=True,
):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_bmi_loss = 0.0
    total_gender_loss = 0.0

    y_true = []
    y_pred = []

    gender_correct = 0
    total_samples = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        bmi = batch["bmi"].to(device)
        gender = batch["gender"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(images)

            bmi_preds = outputs["bmi"]
            gender_logits = outputs["gender_logits"]

            bmi_loss = bmi_criterion(bmi_preds, bmi)
            gender_loss = gender_criterion(gender_logits, gender)

            loss = (
                bmi_loss_weight * bmi_loss
                + gender_loss_weight * gender_loss
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_bmi_loss += bmi_loss.item() * batch_size
        total_gender_loss += gender_loss.item() * batch_size

        y_true.extend(bmi.detach().cpu().numpy())
        y_pred.extend(bmi_preds.detach().cpu().numpy())

        gender_preds = torch.argmax(gender_logits, dim=1)

        gender_correct += (gender_preds == gender).sum().item()
        total_samples += batch_size

    metrics = regression_metrics(y_true, y_pred)

    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["bmi_loss"] = total_bmi_loss / len(loader.dataset)
    metrics["gender_loss"] = total_gender_loss / len(loader.dataset)
    metrics["gender_acc"] = gender_correct / total_samples

    return metrics


def main():
    cfg = load_config()

    set_seed(cfg["project"]["seed"])

    train_cfg = cfg["training"]["resnet50_multitask"]

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

    m2_cfg = cfg["models"]["m2"]

    model = ResNet50MultiTaskBMI(
        hidden_units=tuple(m2_cfg["head_hidden_units"]),
        dropout=m2_cfg["dropout"],
        frozen=m2_cfg["frozen"],
        fine_tune_last_block=m2_cfg["fine_tune_last_block"],
    ).to(device)

    bmi_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    bmi_loss_weight = m2_cfg.get("bmi_loss_weight", 1.0)
    gender_loss_weight = m2_cfg.get("gender_loss_weight", 0.2)

    best_pearson = -999
    history = []

    Path("models").mkdir(exist_ok=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            bmi_criterion=bmi_criterion,
            gender_criterion=gender_criterion,
            optimizer=optimizer,
            device=device,
            bmi_loss_weight=bmi_loss_weight,
            gender_loss_weight=gender_loss_weight,
            train=True,
        )

        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            bmi_criterion=bmi_criterion,
            gender_criterion=gender_criterion,
            optimizer=None,
            device=device,
            bmi_loss_weight=bmi_loss_weight,
            gender_loss_weight=gender_loss_weight,
            train=False,
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "test": test_metrics,
        }

        history.append(record)

        print(
            f"Train MAE: {train_metrics['mae']:.4f} | "
            f"Test MAE: {test_metrics['mae']:.4f} | "
            f"Test RMSE: {test_metrics['rmse']:.4f} | "
            f"Test Pearson r: {test_metrics['pearson_r']:.4f} | "
            f"Test R2: {test_metrics['r2']:.4f} | "
            f"Gender Acc: {test_metrics['gender_acc']:.4f}"
        )

        if test_metrics["pearson_r"] > best_pearson:
            best_pearson = test_metrics["pearson_r"]

            pt_path = "models/resnet50_multitask_best.pt"
            onnx_path = "models/resnet50_multitask_best.onnx"

            torch.save(model.state_dict(), pt_path)
            export_multitask_model_to_onnx(model, onnx_path, device)
            model.train()

            print("Saved new best ResNet50 multi-task model.")
            print(f"Saved PT: {pt_path}")
            print(f"Saved ONNX: {onnx_path}")

    with open(
        "outputs/metrics/resnet50_multitask_history.json",
        "w",
    ) as f:
        json.dump(history, f, indent=2)

    print("\nResNet50 multi-task training complete.")
    print(f"Best Pearson r: {best_pearson:.4f}")


if __name__ == "__main__":
    main()