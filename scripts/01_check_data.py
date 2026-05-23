from pathlib import Path
import sys
import pandas as pd

sys.path.append("src")

from face_bmi.config import load_config


def normalize_image_name(name: str) -> str:
    name = str(name)
    if name.startswith("Images/") or name.startswith("Images\\"):
        name = name.replace("Images/", "").replace("Images\\", "")
    return name


def main():
    cfg = load_config()
    csv_path = Path(cfg["paths"]["raw_csv"])
    images_dir = Path(cfg["paths"]["raw_images_dir"])

    print("Checking dataset...")
    print(f"CSV path: {csv_path}")
    print(f"Images directory: {images_dir}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {csv_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")

    df = pd.read_csv(csv_path)
    print("CSV columns:")
    print(list(df.columns))

    required_cols = [
        cfg["data"]["image_col"],
        cfg["data"]["bmi_col"],
        cfg["data"]["gender_col"],
        cfg["data"]["split_col"],
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"Total rows: {len(df)}")
    print("First 5 rows:")
    print(df.head())

    image_col = cfg["data"]["image_col"]
    df["_image_name"] = df[image_col].apply(normalize_image_name)

    missing_images = []
    for name in df["_image_name"]:
        img_path = images_dir / name
        if not img_path.exists():
            missing_images.append(str(img_path))

    print(f"Found images: {len(df) - len(missing_images)}")
    print(f"Missing images: {len(missing_images)}")

    if missing_images:
        print("First missing image examples:")
        for p in missing_images[:10]:
            print(" -", p)

    bmi_col = cfg["data"]["bmi_col"]
    gender_col = cfg["data"]["gender_col"]
    split_col = cfg["data"]["split_col"]

    print("BMI summary:")
    print(df[bmi_col].describe())

    print("Gender counts:")
    print(df[gender_col].value_counts(dropna=False))

    print("Split counts:")
    print(df[split_col].value_counts(dropna=False))

    if len(missing_images) == 0:
        print("Dataset check passed.")
    else:
        print("Dataset check finished with missing images.")


if __name__ == "__main__":
    main()
