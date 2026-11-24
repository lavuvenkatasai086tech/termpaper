import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from .dataset import create_dataloaders
from .model import EnhancedHybridEfficientNetViT


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix - Enhanced Hybrid EfficientNetV2-B3 + ViT", fontsize=13, pad=15)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {save_path}")


def plot_four_panel_metrics(report_dict, class_names, save_path="evaluation_metrics.png"):
    metrics = ["precision", "recall", "f1-score"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    # Panel 1: Per-Class Precision
    precisions = [report_dict[c]["precision"] * 100 for c in class_names]
    axes[0].bar(class_names, precisions, color="#3b82f6", edgecolor="#1d4ed8")
    axes[0].set_title("Per-Class Precision (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 105)
    axes[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(precisions):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    # Panel 2: Per-Class Recall
    recalls = [report_dict[c]["recall"] * 100 for c in class_names]
    axes[1].bar(class_names, recalls, color="#10b981", edgecolor="#047857")
    axes[1].set_title("Per-Class Recall (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=30)
    for i, v in enumerate(recalls):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    # Panel 3: Per-Class F1-Score
    f1s = [report_dict[c]["f1-score"] * 100 for c in class_names]
    axes[2].bar(class_names, f1s, color="#8b5cf6", edgecolor="#6d28d9")
    axes[2].set_title("Per-Class F1-Score (%)", fontsize=12, fontweight="bold")
    axes[2].set_ylim(0, 105)
    axes[2].tick_params(axis="x", rotation=30)
    for i, v in enumerate(f1s):
        axes[2].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)

    # Panel 4: Class Distribution / Support
    supports = [report_dict[c]["support"] for c in class_names]
    axes[3].bar(class_names, supports, color="#f59e0b", edgecolor="#b45309")
    axes[3].set_title("Test Set Sample Distribution (Support)", fontsize=12, fontweight="bold")
    axes[3].tick_params(axis="x", rotation=30)
    for i, v in enumerate(supports):
        axes[3].text(i, v + 0.5, f"{v}", ha="center", fontsize=9)

    plt.suptitle("Evaluation Benchmarks — Enhanced Hybrid EfficientNetV2-B3 + ViT (Sinamenye et al., 2025)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] 4-panel metrics saved to: {save_path}")


def run_full_evaluation(checkpoint_path="checkpoints/best_model.pth", data_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Compute Device: {device}")

    # Load dataloaders
    train_loader, val_loader, test_loader, class_names, _ = create_dataloaders(
        data_dir=data_dir,
        batch_size=64,
        use_weighted_sampler=False
    )

    model = EnhancedHybridEfficientNetViT(num_classes=len(class_names)).to(device)

    if os.path.exists(checkpoint_path):
        print(f"[Evaluation] Loading weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    else:
        print("[Evaluation Warning] Checkpoint not found; using initialized model.")

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS (Test Split)")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"Macro F1:  {f1 * 100:.2f}%")
    print(f"MCC:       {mcc:.4f}")
    print("=" * 65 + "\n")

    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_four_panel_metrics(rep, class_names)

    report_data = {
        "overall_metrics": {
            "accuracy": float(acc),
            "macro_precision": float(prec),
            "macro_recall": float(rec),
            "macro_f1": float(f1),
            "mcc": float(mcc),
        },
        "per_class": {c: rep[c] for c in class_names if c in rep},
        "published_benchmarks": {
            "paper": "Sinamenye et al. (2025)",
            "accuracy": 0.8506,
            "macro_f1": 0.8377,
            "macro_precision": 0.8286,
            "mcc": 0.8200
        }
    }

    with open("evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    return report_data


if __name__ == "__main__":
    run_full_evaluation()
