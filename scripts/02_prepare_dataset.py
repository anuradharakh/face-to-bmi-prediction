import sys
import json

sys.path.append("src")

from face_bmi.config import load_config
from face_bmi.data.prepare import prepare_dataset


def main():
    cfg = load_config()
    summary = prepare_dataset(cfg)

    print("Dataset preparation complete.\n")
    print(json.dumps(summary, indent=2))

    print("\nCreated files:")
    print(" - data/processed/all_clean.csv")
    print(" - data/processed/train.csv")
    print(" - data/processed/test.csv")
    print(" - outputs/metrics/dataset_summary.json")


if __name__ == "__main__":
    main()