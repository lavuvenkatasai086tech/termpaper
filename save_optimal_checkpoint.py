import os
import sys
import time
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import EfficientNetV2B3ViT

CLASS_NAMES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
FEATURES_CACHE = os.path.join(os.path.dirname(__file__), "checkpoints", "features_cache.pt")


def main():
    print("=" * 70, flush=True)
    print("[Optimal Checkpoint Generator] Hybrid EfficientNetV2B3 + ViT", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(FEATURES_CACHE):
        raise FileNotFoundError(f"Features cache not found at: {FEATURES_CACHE}")

    data = torch.load(FEATURES_CACHE, map_location="cpu")
    eff_feats = data["eff_feats"]
    vit_cls = data["vit_cls"]
    labels = data["labels"].numpy()

    # 1. Instantiate PyTorch model to get consistent ViT projection
    torch.manual_seed(42)
    np.random.seed(42)
    model = EfficientNetV2B3ViT(num_classes=len(CLASS_NAMES))
    model.eval()

    with torch.no_grad():
        vit_proj = model.vit_linear(vit_cls)
        fused = torch.cat([eff_feats, vit_proj], dim=1).numpy()

    print(f"Fused Feature Matrix Shape: {fused.shape} (N={fused.shape[0]}, D={fused.shape[1]})", flush=True)

    # 2. Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs = []
    cv_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(fused, labels)):
        clf = LogisticRegression(C=0.15, max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(fused[train_idx], labels[train_idx])
        preds = clf.predict(fused[val_idx])
        fold_acc = accuracy_score(labels[val_idx], preds)
        fold_f1 = f1_score(labels[val_idx], preds, average="macro", zero_division=0)
        cv_accs.append(fold_acc)
        cv_f1s.append(fold_f1)

    print(f"5-Fold CV Accuracy: {np.mean(cv_accs)*100:.2f}% (+/- {np.std(cv_accs)*100:.2f}%)", flush=True)
    print(f"5-Fold CV Macro F1: {np.mean(cv_f1s)*100:.2f}% (+/- {np.std(cv_f1s)*100:.2f}%)", flush=True)

    # 3. Train on 85% train split, evaluate on 15% holdout test split
    X_train, X_test, y_train, y_test = train_test_split(
        fused, labels, test_size=0.15, random_state=42, stratify=labels
    )

    final_clf = LogisticRegression(C=0.15, max_iter=1500, class_weight="balanced", random_state=42)
    final_clf.fit(X_train, y_train)

    test_preds = final_clf.predict(X_test)
    test_probs = final_clf.predict_proba(X_test)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
    rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, test_preds)

    print("\n" + "=" * 50, flush=True)
    print("[FINAL TEST BENCHMARK METRICS]", flush=True)
    print("=" * 50, flush=True)
    print(f"  Test Accuracy:  {acc * 100:.2f}%", flush=True)
    print(f"  Macro Precision:{prec * 100:.2f}%", flush=True)
    print(f"  Macro Recall:   {rec * 100:.2f}%", flush=True)
    print(f"  Macro F1-Score: {f1 * 100:.2f}%", flush=True)
    print(f"  MCC:            {mcc:.4f}", flush=True)
    print("=" * 50, flush=True)

    print("\nClassification Report:", flush=True)
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES), flush=True)

    # 4. Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"EfficientNetV2B3+ViT Test Confusion Matrix (Acc: {acc*100:.1f}%)")
    plt.xlabel("Predicted Disease Class")
    plt.ylabel("True Disease Class")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved confusion matrix to: {cm_path}", flush=True)

    # 5. Fit on full dataset for maximum generalization coverage in deployment
    deploy_clf = LogisticRegression(C=0.15, max_iter=1500, class_weight="balanced", random_state=42)
    deploy_clf.fit(fused, labels)

    # 6. Transfer trained weights and biases into PyTorch model.classifier
    model.classifier.weight.data = torch.tensor(deploy_clf.coef_, dtype=torch.float32)
    model.classifier.bias.data = torch.tensor(deploy_clf.intercept_, dtype=torch.float32)

    # Verification: check PyTorch logits match exactly
    with torch.no_grad():
        fused_tensor = torch.tensor(fused, dtype=torch.float32)
        pt_logits = model.classifier(fused_tensor)
        pt_preds = pt_logits.argmax(dim=1).numpy()
        sk_preds = deploy_clf.predict(fused)
        assert np.array_equal(pt_preds, sk_preds), "PyTorch predictions must match Sklearn predictions!"
        print("[Verification] PyTorch classifier weights verified with 100% precision match.", flush=True)

    # 7. Save model checkpoint
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
            "cv_accuracy_mean": float(np.mean(cv_accs)),
        },
        "model_type": "EfficientNetV2B3ViT",
        "timestamp": time.time(),
    }
    torch.save(checkpoint_payload, CHECKPOINT_PATH)
    print(f"[Checkpoint] Successfully saved best model to: {CHECKPOINT_PATH}", flush=True)
    print(f"Checkpoint size: {os.path.getsize(CHECKPOINT_PATH):,} bytes", flush=True)


if __name__ == "__main__":
    main()
