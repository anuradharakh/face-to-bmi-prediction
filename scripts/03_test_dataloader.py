import sys

sys.path.append("src")

from torch.utils.data import DataLoader

from face_bmi.config import load_config
from face_bmi.data.dataset import FaceBMIDataset


def main():
    cfg = load_config()

    image_size = cfg["data"]["image_size"]
    batch_size = cfg["training"]["batch_size"]

    train_csv = "data/processed/train.csv"
    test_csv = "data/processed/test.csv"

    train_dataset = FaceBMIDataset(
        csv_path=train_csv,
        image_size=image_size,
        is_train=True,
    )

    test_dataset = FaceBMIDataset(
        csv_path=test_csv,
        image_size=image_size,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    print("Dataloader test passed.")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image batch shape: {batch['image'].shape}")
    print(f"BMI batch shape: {batch['bmi'].shape}")
    print(f"Gender batch shape: {batch['gender'].shape}")
    print(f"Example image path: {batch['image_path'][0]}")


if __name__ == "__main__":
    main()