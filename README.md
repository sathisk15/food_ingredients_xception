# Image → Ingredients Multi-Label Classification (Xception from Scratch)

This project implements an **image-to-ingredients multi-label classification system** using a **from-scratch Xception architecture** in **pure PyTorch**.

The model predicts a set of ingredients given a food image, trained on the Kaggle  
**Food Ingredients and Recipe Dataset with Images**.

---

## 🔒 Constraints (Strictly Followed)

- **Xception implemented completely from scratch**
  - No pretrained weights
  - No `torchvision.models`
  - No `timm`, Keras, HuggingFace, etc.
- **Pure PyTorch + standard Python libraries**
- **Multi-label classification**
  - Sigmoid outputs
  - `BCEWithLogitsLoss`
- **Cross-platform**
  - macOS (Apple Silicon, MPS backend)
  - Windows (CUDA or CPU) without code changes

---

## 📁 Project Structure

```
food_ingredients_xception/
│
├── data/
│   ├── raw/
│   │   ├── Food Images/
│   │   └── Food Ingredients and Recipe Dataset with Image Name Mapping.csv
│   └── processed/
│       ├── ingredient_vocab.json
│       ├── ingredient_vocab_pruned.json
│       ├── labels.npy
│       └── labels_pruned.npy
│
├── src/
│   ├── datasets/
│   │   └── food_dataset.py
│   ├── models/
│   │   ├── layers.py
│   │   └── xception.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── utils/
│   │   └── device.py
│   └── inference.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── encode_labels.py
│   └── prune_labels.py
│
├── run_train.py
├── checkpoints/
├── checkpoints_pruned/
└── README.md
```

---

## 🧠 Model Architecture

- **Xception (Chollet-style)**
  - Depthwise separable convolutions
  - Residual connections
  - Entry / Middle / Exit flows
- **Adaptive Average Pooling**
- **Fully connected classifier**
- **No sigmoid inside the model**
  - Sigmoid applied only for evaluation/inference

---

## 🏷️ Label Strategy

### Initial Labeling

- Ingredients parsed directly from recipe text
- Minimal cleaning
- Top **1000 most frequent labels**

### Pruning

- Labels appearing in `< 50` samples removed
- Final label space: **~400 ingredients**
- Statistical pruning only (no NLP tricks)

---

## 🚀 Training

### Device Handling

Automatically selects:

1. MPS (Apple Silicon)
2. CUDA
3. CPU

### Loss & Optimizer

- `BCEWithLogitsLoss`
- `AdamW`
- Gradient clipping (`1.0`)
- Batch size: `16` (MPS-safe)

### Run Training

```bash
python run_train.py
```

---

## 📊 Evaluation

Metrics:

- Micro Precision / Recall / F1
- Macro F1

---

## 🔍 Inference (Single Image)

```bash
python - << 'EOF'
from src.inference import predict_ingredients

preds = predict_ingredients(
    image_path="data/raw/Food Images/example.jpg",
    checkpoint_path="checkpoints_pruned/xception_epoch_10.pt",
    vocab_path="data/processed/ingredient_vocab_pruned.json",
    top_k=10,
)

for ing, score in preds:
    print(ing, score)
EOF
```

---

## ⚠️ Known Limitations

- Some non-visual labels may still exist if they are frequent
- Recipe text contains noise (instructions, quantities)
- No validation split
- No data augmentation

---

## 📌 Future Improvements

- Manual blacklist of non-visual labels
- Validation split + threshold calibration
- Class imbalance weighting
- Data augmentation

---

## ✍️ Author

**Sathiskumar**  
MSc Applied Computer Science
