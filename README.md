# Face-to-BMI Final Project

A configurable, step-by-step project repository for BMI prediction from face images.

## Our finalized model plan

| Model | Type | Goal |
|---|---|---|
| M1 | VGG16 ImageNet frozen backbone + 1–2 dense layers | Baseline feature extraction |
| M2 | ResNet50 multi-task model: BMI regression + gender classification | Main model to beat paper |
| M3 | EfficientNet fine-tuning with last blocks unfrozen | Advanced fine-tuning model |

## Local Mac setup

This project is designed to run locally on your MacBook Pro with 96GB RAM.

We will use:
- PyTorch
- Apple MPS when available
- YAML configuration files
- Streamlit frontend

## Expected data format

Place your files like this:

```text
data/raw/data.csv
data/raw/Images/
```

CSV expected columns:

```text
bmi, gender, is_training, name
```

Example image path:

```text
data/raw/Images/img_0.bmp
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 1 command

After placing the dataset:

```bash
python scripts/01_check_data.py
```

This verifies:
- CSV exists
- image folder exists
- required columns exist
- image files can be found
- train/test split is readable
