from pathlib import Path
import json
import pandas as pd


def normalize_image_name(name: str) -> str:
    name = str(name)
    name = name.replace("\\", "/")
    if name.startswith("Images/"):
        name = name.replace("Images/", "", 1)
    return name


def prepare_dataset(cfg: dict) -> dict:
    csv_path = Path(cfg["paths"]["raw_csv"])
    images_dir = Path(cfg["paths"]["raw_images_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    metrics_dir = Path(cfg["paths"]["metrics_dir"])

    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    image_col = cfg["data"]["image_col"]
    bmi_col = cfg["data"]["bmi_col"]
    gender_col = cfg["data"]["gender_col"]
    split_col = cfg["data"]["split_col"]
    gender_map = cfg["data"]["gender_map"]

    df = pd.read_csv(csv_path)

    df = df.copy()
    df["image_name"] = df[image_col].apply(normalize_image_name)
    df["image_path"] = df["image_name"].apply(lambda x: str(images_dir / x))
    df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())

    before_count = len(df)
    missing_count = int((~df["image_exists"]).sum())

    clean = df[df["image_exists"]].copy()
    clean = clean.drop(columns=["image_exists"])

    clean["bmi"] = pd.to_numeric(clean[bmi_col], errors="coerce")
    clean = clean.dropna(subset=["bmi", gender_col, split_col, "image_path"])

    clean["gender_encoded"] = clean[gender_col].map(gender_map)
    clean["gender_encoded"] = clean["gender_encoded"].astype(int)
    clean["is_training"] = clean[split_col].astype(int)

    train = clean[clean["is_training"] == 1].copy()
    test = clean[clean["is_training"] == 0].copy()

    clean_out = processed_dir / "all_clean.csv"
    train_out = processed_dir / "train.csv"
    test_out = processed_dir / "test.csv"

    clean.to_csv(clean_out, index=False)
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)

    summary = {
        "rows_before_cleaning": before_count,
        "missing_images": missing_count,
        "rows_after_cleaning": int(len(clean)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "gender_counts": clean[gender_col].value_counts().to_dict(),
        "outputs": {
            "all_clean": str(clean_out),
            "train": str(train_out),
            "test": str(test_out),
        },
    }

    summary_path = metrics_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary