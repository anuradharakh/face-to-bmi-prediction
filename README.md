
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
