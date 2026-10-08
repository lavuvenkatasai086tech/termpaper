import os
import sys
import time
import random
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import EfficientNetV2B3ViT
from dataset import get_transforms

CLASS_NAMES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "data",
    "potatodata",
    "Potato Leaf Disease Dataset in Uncontrolled Environment"
)
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
FEATURES_CACHE = os.path.join(os.path.dirname(__file__), "checkpoints", "features_cache.pt")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_dataset(max_per_class=120):
    """
    Collects a balanced stratified sample of image paths and labels.
    """
    samples = []
    print(f"[Dataset] Scanning images from: {DATA_DIR}")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_folder):
            continue
        all_files = [
            os.path.join(class_folder, f)
            for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.shuffle(all_files)
        selected = all_files[:max_per_class] if max_per_class else all_files
        print(f"  - {class_name:12s}: {len(selected)} images selected (total: {len(all_files)})")
        for img_path in selected:
            samples.append((img_path, class_idx))

    random.shuffle(samples)
    print(f"[Dataset] Total samples selected: {len(samples)}")
    return samples


def extract_features(model, samples, batch_size=32, device="cpu"):
    """
    Extracts frozen EfficientNet and ViT features in batches.
    """
    if os.path.exists(FEATURES_CACHE):
        print(f"[Cache] Loading cached extracted features from: {FEATURES_CACHE}")
        data = torch.load(FEATURES_CACHE, map_location="cpu")
        return data["eff_feats"], data["vit_cls"], data["labels"]

    print(f"[Extraction] Extracting features with batch size {batch_size} on {device}...")
    torch.set_num_threads(os.cpu_count() or 4)
    model.eval()

    transform = get_transforms(augment=False)
    all_eff_feats = []
    all_vit_cls = []
    all_labels = []

    total_batches = (len(samples) + batch_size - 1) // batch_size
    t_start = time.time()

    for b_idx in range(total_batches):
        batch_slice = samples[b_idx * batch_size : (b_idx + 1) * batch_size]
        batch_tensors = []
        batch_y = []

        for img_path, label in batch_slice:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(transform(img))
                batch_y.append(label)
            except Exception as e:
                print(f"  Warning: failed to read {img_path}: {e}")

        if not batch_tensors:
            continue

        x = torch.stack(batch_tensors).to(device)

        with torch.inference_mode():
            # EffNet GAP features (shape: B, 1536)
            eff_out = model.effnet(x)

            # ViT CLS token (shape: B, 768)
            vit_out = model.vit(x)
            vit_cls = vit_out.last_hidden_state[:, 0, :]

        all_eff_feats.append(eff_out.cpu())
        all_vit_cls.append(vit_cls.cpu())
        all_labels.extend(batch_y)

        elapsed = time.time() - t_start
        done_imgs = min((b_idx + 1) * batch_size, len(samples))
        rate = done_imgs / max(elapsed, 0.001)
        eta = (len(samples) - done_imgs) / max(rate, 0.001)
        print(f"  Batch {b_idx + 1}/{total_batches} [{done_imgs}/{len(samples)}] - {rate:.2f} img/s - ETA: {eta:.0f}s", flush=True)

    eff_feats_tensor = torch.cat(all_eff_feats, dim=0)
    vit_cls_tensor = torch.cat(all_vit_cls, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    os.makedirs(os.path.dirname(FEATURES_CACHE), exist_ok=True)
    torch.save({
        "eff_feats": eff_feats_tensor,
        "vit_cls": vit_cls_tensor,
        "labels": labels_tensor
    }, FEATURES_CACHE)
    print(f"[Cache] Extracted features saved to: {FEATURES_CACHE}", flush=True)

    return eff_feats_tensor, vit_cls_tensor, labels_tensor


def train_head():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("[Hybrid Head Training] Potato Leaf Disease Classification", flush=True)
    print(f"   Device: {device}", flush=True)
    print("=" * 70, flush=True)

    # 1. Instantiate the model
    print("[Model] Initializing EfficientNetV2B3ViT...", flush=True)
    model = EfficientNetV2B3ViT(num_classes=len(CLASS_NAMES))
    model.to(device)

    # 2. Collect dataset samples
    samples = collect_dataset(max_per_class=100)

    # 3. Extract or load features
    eff_feats, vit_cls, labels = extract_features(model, samples, batch_size=32, device=device)

    num_samples = len(labels)
    indices = list(range(num_samples))
    random.shuffle(indices)

    # Stratified or randomized 80/10/10 split
    n_train = int(num_samples * 0.80)
    n_val = int(num_samples * 0.10)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_ds = TensorDataset(eff_feats[train_idx], vit_cls[train_idx], labels[train_idx])
    val_ds = TensorDataset(eff_feats[val_idx], vit_cls[val_idx], labels[val_idx])
    test_ds = TensorDataset(eff_feats[test_idx], vit_cls[test_idx], labels[test_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"\n[Split] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # 4. Trainable Head Module
    class HybridClassifierHead(nn.Module):
        def __init__(self, vit_linear, dropout, classifier):
            super().__init__()
            self.vit_linear = vit_linear
            self.dropout = dropout
            self.classifier = classifier

        def forward(self, eff_f, vit_c):
            vit_proj = self.vit_linear(vit_c)
            fused = torch.cat([eff_f, vit_proj], dim=1)
            logits = self.classifier(self.dropout(fused))
            return logits

    head = HybridClassifierHead(model.vit_linear, model.dropout, model.classifier).to(device)

    # Calculate class weights for balance
    train_labels = labels[train_idx].numpy()
    class_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    total_train = len(train_labels)
    class_weights = total_train / (len(CLASS_NAMES) * np.maximum(class_counts, 1).astype(float))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    epochs = 40
    best_val_acc = 0.0
    best_state = None

    print("\n[Training] Optimizing classification head and ViT projection...")
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for b_eff, b_vit, b_y in train_loader:
            b_eff, b_vit, b_y = b_eff.to(device), b_vit.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = head(b_eff, b_vit)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * b_eff.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == b_y).sum().item()
            total += b_eff.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for b_eff, b_vit, b_y in val_loader:
                b_eff, b_vit, b_y = b_eff.to(device), b_vit.to(device), b_y.to(device)
                logits = head(b_eff, b_vit)
                val_correct += (logits.argmax(dim=1) == b_y).sum().item()
                val_total += b_eff.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}%")

    print(f"\n[Validation] Peak Validation Accuracy: {best_val_acc * 100:.2f}%")

    # Load best state back into model
    if best_state is not None:
        model.load_state_dict(best_state)

    # 5. Final Evaluation on Test Set
    head = HybridClassifierHead(model.vit_linear, model.dropout, model.classifier).to(device)
    head.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for b_eff, b_vit, b_y in test_loader:
            b_eff, b_vit, b_y = b_eff.to(device), b_vit.to(device), b_y.to(device)
            logits = head(b_eff, b_vit)
            test_preds.extend(logits.argmax(dim=1).cpu().numpy())
            test_targets.extend(b_y.cpu().numpy())

    y_true = np.array(test_targets)
    y_pred = np.array(test_preds)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n" + "=" * 50)
    print("[TEST SET BENCHMARK RESULTS]")
    print("=" * 50)
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {prec * 100:.2f}%")
    print(f"  Recall:    {rec * 100:.2f}%")
    print(f"  Macro F1:  {f1 * 100:.2f}%")
    print(f"  MCC:       {mcc:.4f}")
    print("=" * 50)

    # Save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Hybrid Model Test Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.xlabel("Predicted Disease Class")
    plt.ylabel("Ground Truth Disease Class")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved confusion matrix to: {cm_path}")

    # 6. Save model checkpoint
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "metrics": {
            "test_accuracy": acc,
            "test_precision": prec,
            "test_recall": rec,
            "test_f1": f1,
            "test_mcc": mcc,
        },
        "model_type": "EfficientNetV2B3ViT",
        "timestamp": time.time(),
    }
    torch.save(checkpoint_payload, CHECKPOINT_PATH)
    print(f"[Checkpoint] Successfully saved best model to: {CHECKPOINT_PATH}")
    print(f"Checkpoint file size: {os.path.getsize(CHECKPOINT_PATH)} bytes")


if __name__ == "__main__":
    train_head()
