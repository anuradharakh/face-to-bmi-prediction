from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_train_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_eval_transforms(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class FaceBMIDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 224,
        is_train: bool = True,
        transform: Optional[transforms.Compose] = None,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        self.image_size = image_size
        self.is_train = is_train

        if transform is None:
            self.transform = (
                get_train_transforms(image_size)
                if is_train
                else get_eval_transforms(image_size)
            )
        else:
            self.transform = transform

        required_cols = ["image_path", "bmi", "gender_encoded"]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        bmi = torch.tensor(float(row["bmi"]), dtype=torch.float32)
        gender = torch.tensor(int(row["gender_encoded"]), dtype=torch.long)

        return {
            "image": image,
            "bmi": bmi,
            "gender": gender,
            "image_path": str(image_path),
        }