# Potato Plant Disease Detection: Hybrid EfficientNetV2B3 + ViT (Sinamenye et al., 2025)

A clean, modular, and robust PyTorch implementation of the hybrid deep learning architecture **EfficientNetV2B3 + ViT** for potato plant disease detection, following the exact specifications from the research paper:
> **"Potato plant disease detection: leveraging hybrid deep learning models"** (Sinamenye et al., 2025, *BMC Plant Biology*).

---

## 1. Architecture Overview

The model employs a parallel dual-path feature extraction strategy that combines local convolutional features with global contextual attention:

```
                          Input RGB Image (224x224)
                                    |
                 +------------------+------------------+
                 |                                     |
           [CNN Branch]                          [ViT Branch]
      EfficientNetV2B3 Backbone             ViT-Base (16x16 patch)
         (Weights Frozen)                      (Weights Frozen)
                 |                                     |
        GAP / Feature Flatten                 [CLS] Token (768-d)
                 |                                     |
        Local Vector (1536-d)                Linear Projection (512-d)
                 |                                     |
                 +------------------+------------------+
                                    |
                        Concatenation Layer (2048-d)
                                    |
                            Dropout (rate = 0.2)
                                    |
                        Dense Linear Layer (7 classes)
                                    |
                           Output Logits / Softmax
```

### 7 Target Classes:
1. **Bacteria**
2. **Fungi**
3. **Healthy**
4. **Nematode**
5. **Pest**
6. **Phytophthora**
7. **Virus**

---

## 2. Improvements Over the Existing Repository (`potato-efficientViT-main`)

1. **Fixed PyTorch Dataset Transform Bug:**
   - In the previous codebase, `train_dataset.dataset.transform` was modified on the same shared `ImageFolder` instance as validation and test subsets. As a result, the subsequent assignment `vali_dataset.dataset.transform = self.other_transforms` overwrote the training transformations, disabling data augmentations during training.
   - Fixed via `PotatoLeafDataset` which decouples subset sample indices and applies individual transformation pipelines independently.

2. **Stratified Splitting:**
   - Imbalanced classes (e.g., Nematode with only 68 samples vs. Fungi with 748) are preserved with exact class proportions across train (80%), validation (10%), and test (10%) splits.

3. **No Mandatory WandB Hard-Crash:**
   - The original code called `wandb.login()` at import time, crashing immediately if no API key or `.env` file was present.
   - Now, console logging and local metric tracking work out-of-the-box, with an optional `--wandb` flag for users who want cloud tracking.

4. **Checkpointing & Metrics:**
   - Now tracks validation loss, saves both `best_model.pth` and `last_model.pth`, computes Accuracy, Macro Precision, Macro Recall, Macro F1, and MCC, and generates publication-ready confusion matrix plots.

---

## 3. Directory Structure

```
Termpaper Project/
├── dataset.py                                                 # Augmentation pipelines, custom dataset & DataLoaders
├── model.py                                                   # EfficientNetV2B3 + ViT hybrid architecture
├── train.py                                                   # Training loop, scheduler, metrics & evaluation
├── predict.py                                                 # Single-image and batch inference script
├── data/
│   └── potatodata/
│       └── Potato Leaf Disease Dataset in Uncontrolled Environment/
│           ├── Bacteria/
│           ├── Fungi/
│           ├── Healthy/
│           ├── Nematode/
│           ├── Pest/
│           ├── Phytopthora/
│           └── Virus/
├── checkpoints/                                               # Saved model weights & metrics (auto-created)
│   ├── best_model.pth
│   ├── last_model.pth
│   └── test_confusion_matrix.png
└── README.md
```

---

## 4. Hyperparameter Configuration (Paper Specifications)

| Hyperparameter | Value | Description |
|---|---|---|
| **Input Image Size** | 256x256 -> 224x224 | Initial resize to 256, RandomResizedCrop to 224 |
| **Augmentation** | Flips, Affine, ColorJitter | Rotation (-40° to +40°), saturation 0.8, hue 0.021, zoom 0.8-1.2, translation 0.13 |
| **Normalization** | ImageNet Mean & Std | Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] |
| **Backbones** | EfficientNetV2B3 & ViT-Base | Both backbones frozen, only projection & classifier head trained |
| **Dropout Rate** | 0.2 | Applied prior to classification dense layer |
| **Batch Size** | 64 | Configurable via `--batch_size` |
| **Epochs** | 70 | Configurable via `--epochs` |
| **Optimizer** | Adam (lr = 1e-4) | Applied to trainable parameters |
| **Scheduler** | ReduceLROnPlateau | Patience: 5 epochs, Factor: 0.5 based on validation loss |
| **Loss Function** | Cross-Entropy Loss | Logits input |

---

## 5. Usage

### A. Quick Sanity Test (1 epoch, small batch)
```bash
python train.py --quick_test
```

### B. Full Training (70 Epochs)
```bash
python train.py --epochs 70 --batch_size 64 --lr 0.0001
```

### C. Evaluate an Existing Checkpoint
```bash
python train.py --eval_only checkpoints/best_model.pth
```

### D. Run Inference on a Single Leaf Image
```bash
python predict.py --image path/to/leaf.jpg --checkpoint checkpoints/best_model.pth
```
