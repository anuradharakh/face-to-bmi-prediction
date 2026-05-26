
# Face-to-BMI Prediction using Deep Learning

Deep learning project for predicting BMI from facial images using transfer learning, multi-task learning, and fine-tuning strategies.

## Final Results

| Model | Description | Pearson r |
|---|---|---|
| M1 | VGG16 Baseline | 0.4227 |
| M3 | EfficientNet Fine-Tune | 0.5532 |
| M2B | ResNet50 BMI-Only | 0.5771 |
| M2 | ResNet50 Multi-Task | **0.6811** |

## Best Model

ResNet50 Multi-Task Learning achieved:

- Pearson Correlation = 0.6811
- Surpassed paper benchmark of 0.6500

---

## Models

### M1 — VGG16 Baseline
- Frozen VGG16 ImageNet backbone
- Dense regression head
- Transfer learning baseline

Result:
- Pearson r = 0.4227

---

### M2 — ResNet50 Multi-Task
Architecture:
- Shared ResNet50 backbone
- BMI regression head
- Gender classification head

Improvements:
- Fine-tuned last ResNet block
- Lower learning rate
- Reduced gender loss weight
- Longer training
- Better regularization

Final Result:
- Pearson r = 0.6811

---

### M3 — EfficientNet Fine-Tuning
Improvements:
- Unfroze last 3 EfficientNet blocks
- Reduced learning rate
- Fine-tuning strategy

Final Result:
- Pearson r = 0.5532

---

### M2B — ResNet50 BMI-Only
Purpose:
- Ablation study removing gender classification

Observation:
- Removing gender slightly reduced performance

Final Result:
- Pearson r = 0.5771

---

## Streamlit Frontend

Features:
- Upload image
- Webcam capture
- Toggle models on/off
- Multi-model comparison
- Loading spinner
- Clear/reset buttons
- Dark modern UI

Run:

streamlit run app/streamlit_app.py

---

## Key Concepts

- Transfer learning
- Fine-tuning
- Multi-task learning
- CNN feature extraction
- Regression
- Data augmentation
- Hyperparameter tuning
- Streamlit deployment

---

## Conclusion

The final ResNet50 multi-task model achieved:

Pearson r = 0.6811

which surpassed the original research benchmark.


# Run Training Scripts

## Step 1 — Activate Virtual Environment

```bash
source .venv/bin/activate
```

---

## Step 2 — Data Preparation Pipeline

## Check Dataset

```bash
python scripts/01_check_data.py
```

Validates:
- CSV structure
- missing images
- dataset statistics

---

## Prepare Dataset

```bash
python scripts/02_prepare_dataset.py
```

Creates cleaned train/test CSV files.

---

## Test Dataloader

```bash
python scripts/03_test_dataloader.py
```

Verifies:
- image loading
- transforms
- tensor shapes

---

## Test Metrics

```bash
python scripts/04_test_metrics.py
```

Tests:
- MAE
- RMSE
- Pearson correlation
- R² score

---

# Step 3 — Train Models

## M1 — VGG16 Baseline

```bash
python scripts/05_train_m1_vgg16.py
```

Output:
- `models/m1_vgg16_best.pt`
- `models/m1_vgg16_best.onnx`

---

## M2 — ResNet50 Multi-Task (Best Model)

```bash
python scripts/06_train_m2_resnet50_multitask.py
```

Output:
- `models/resnet50_multitask_best.pt`
- `models/resnet50_multitask_best.onnx`

Best result:

```text
Pearson r = 0.6811
```

---

## M3 — EfficientNet Fine-Tuning

```bash
python scripts/07_train_efficientnet_finetune.py
```

Output:
- `models/efficientnet_finetune_best.pt`
- `models/efficientnet_finetune_best.onnx`

---

## M2B — ResNet50 BMI-Only

```bash
python scripts/08_train_resnet50_bmi_only.py
```

Output:
- `models/resnet50_bmi_only_best.pt`
- `models/resnet50_bmi_only_best.onnx`

---

# Step 4 — Run Streamlit Application

```bash
streamlit run app/streamlit_app.py
```

Features:
- image upload
- webcam capture
- model selection
- prediction comparison
- loading spinner
- reset/clear functionality

---

# Optional — Run Multiple Models in Parallel

Example:

## Terminal 1

```bash
python scripts/06_train_m2_resnet50_multitask.py
```

## Terminal 2

```bash
python scripts/07_train_efficientnet_finetune.py
```

Recommended for Apple Silicon Macs with high memory.