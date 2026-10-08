import os
import sys
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    log_loss,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import EfficientNetV2B3ViT

CLASS_NAMES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
FEATURES_CACHE = os.path.join(os.path.dirname(__file__), "checkpoints", "features_cache.pt")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_metrics.png")


def run_comprehensive_evaluation():
    print("=" * 75, flush=True)
    print("🌿 HYBRID DEEP LEARNING MODEL: COMPREHENSIVE EVALUATION BENCHMARK", flush=True)
    print("   Architecture: EfficientNetV2B3 + Vision Transformer (Sinamenye et al., 2025)", flush=True)
    print(f"   Checkpoint:   {CHECKPOINT_PATH}", flush=True)
    print("=" * 75, flush=True)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    # Check for full dataset features cache first
    full_cache = os.path.join(os.path.dirname(__file__), "checkpoints", "features_cache_full.pt")
    cache_path = full_cache if os.path.exists(full_cache) else FEATURES_CACHE
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Features cache not found at: {cache_path}")

    # 1. Load model and checkpoint
    device = torch.device("cpu")
    model = EfficientNetV2B3ViT(num_classes=len(CLASS_NAMES))
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load dataset features
    print(f"[Dataset] Loading features from: {cache_path}", flush=True)
    cache_data = torch.load(cache_path, map_location=device)
    eff_feats = cache_data["eff_feats"].float()
    vit_cls = cache_data["vit_cls"].float()
    labels = cache_data["labels"].long().numpy()

    # Stratified Train/Test split: 90% Train, 10% Test (as in Sinamenye et al., 2025)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.10, random_state=42, stratify=labels)

    # Evaluate the trained PyTorch hybrid model directly on unseen test split
    with torch.no_grad():
        vp = model.vit_linear(vit_cls[test_idx])
        fused_test = torch.cat([eff_feats[test_idx], vp], dim=1)
        test_logits = model.classifier(fused_test)
        y_prob = F.softmax(test_logits, dim=1).numpy()
        y_pred = test_logits.argmax(dim=1).numpy()

    y_test = labels[test_idx]

    # 3. Overall Metrics
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    ce_loss = log_loss(y_test, y_prob)

    # Multi-class ROC AUC (One-vs-Rest)
    y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
    roc_auc_ovr = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")
    roc_auc_weighted = roc_auc_score(y_test_bin, y_prob, average="weighted", multi_class="ovr")

    # 4. Per-Class Metrics Calculation (Sensitivity, Specificity, Precision, Recall, F1)
    cm = confusion_matrix(y_test, y_pred)
    total_test = len(y_test)
    per_class_stats = []

    print("\n" + "-" * 75, flush=True)
    print(f"{'Class Name':14s} | {'Support':7s} | {'Prec':6s} | {'Rec/Sens':8s} | {'Spec':6s} | {'F1-Score':8s} | {'AUC':6s}", flush=True)
    print("-" * 75, flush=True)

    for i, c_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total_test - (tp + fp + fn)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_val = 2 * (p_val * sens) / (p_val + sens) if (p_val + sens) > 0 else 0.0
        
        try:
            auc_val = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
        except Exception:
            auc_val = 0.5

        per_class_stats.append({
            "class": c_name,
            "support": int(cm[i, :].sum()),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": round(float(p_val), 4),
            "recall_sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "f1_score": round(float(f1_val), 4),
            "roc_auc": round(float(auc_val), 4),
        })

        print(f"{c_name:14s} | {cm[i, :].sum():7d} | {p_val*100:5.1f}% | {sens*100:7.1f}% | {spec*100:5.1f}% | {f1_val*100:7.1f}% | {auc_val:5.3f}", flush=True)

    print("-" * 75, flush=True)

    # 5. Print Overall Summary
    print("\n" + "=" * 50, flush=True)
    print("📈 SUMMARY OF OVERALL EVALUATION METRICS", flush=True)
    print("=" * 50, flush=True)
    print(f"  Test Accuracy:              {acc * 100:.2f}%", flush=True)
    print(f"  Macro Precision:            {prec_macro * 100:.2f}%", flush=True)
    print(f"  Macro Recall / Sensitivity: {rec_macro * 100:.2f}%", flush=True)
    print(f"  Macro F1-Score:             {f1_macro * 100:.2f}%", flush=True)
    print(f"  Weighted F1-Score:          {f1_weighted * 100:.2f}%", flush=True)
    print(f"  Matthews Corr. Coeff (MCC): {mcc:.4f}", flush=True)
    print(f"  Cohen's Kappa (κ):          {kappa:.4f}", flush=True)
    print(f"  ROC-AUC (Macro OVR):        {roc_auc_ovr:.4f}", flush=True)
    print(f"  ROC-AUC (Weighted OVR):     {roc_auc_weighted:.4f}", flush=True)
    print(f"  Cross-Entropy Log Loss:     {ce_loss:.4f}", flush=True)
    print("=" * 50, flush=True)

    # 6. Save JSON report
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "EfficientNetV2B3 + ViT (Sinamenye et al., 2025)",
        "test_samples_count": total_test,
        "overall_metrics": {
            "accuracy": round(float(acc), 4),
            "macro_precision": round(float(prec_macro), 4),
            "weighted_precision": round(float(prec_weighted), 4),
            "macro_recall": round(float(rec_macro), 4),
            "weighted_recall": round(float(rec_weighted), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "mcc": round(float(mcc), 4),
            "cohens_kappa": round(float(kappa), 4),
            "roc_auc_macro_ovr": round(float(roc_auc_ovr), 4),
            "roc_auc_weighted_ovr": round(float(roc_auc_weighted), 4),
            "cross_entropy_loss": round(float(ce_loss), 4),
        },
        "per_class_metrics": per_class_stats,
        "confusion_matrix": cm.tolist(),
        "classes": CLASS_NAMES,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Artifact] Saved detailed metrics report JSON: {REPORT_PATH}", flush=True)

    # 7. Generate 4-Panel Visualization Plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Normalized Confusion Matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0, 0],
        cbar=True,
    )
    axes[0, 0].set_title("1. Normalized Confusion Matrix (Sensitivity per Class)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Predicted Class", fontsize=10)
    axes[0, 0].set_ylabel("True Class", fontsize=10)
    axes[0, 0].tick_params(axis="x", rotation=35)

    # Panel 2: Per-Class Precision & Recall Comparison
    bar_width = 0.35
    x_indices = np.arange(len(CLASS_NAMES))
    p_vals = [s["precision"] * 100 for s in per_class_stats]
    r_vals = [s["recall_sensitivity"] * 100 for s in per_class_stats]

    axes[0, 1].bar(x_indices - bar_width/2, p_vals, width=bar_width, label="Precision", color="#0ea5e9", alpha=0.9)
    axes[0, 1].bar(x_indices + bar_width/2, r_vals, width=bar_width, label="Recall (Sensitivity)", color="#10b981", alpha=0.9)
    axes[0, 1].set_xticks(x_indices)
    axes[0, 1].set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    axes[0, 1].set_ylabel("Percentage (%)", fontsize=10)
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_title("2. Precision vs. Recall by Pathology", fontsize=12, fontweight="bold")
    axes[0, 1].legend(loc="upper right")

    # Panel 3: Multi-Class ROC Curves
    palette = sns.color_palette("husl", len(CLASS_NAMES))
    for i, c_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        auc_val = per_class_stats[i]["roc_auc"]
        axes[1, 0].plot(fpr, tpr, label=f"{c_name} (AUC={auc_val:.2f})", color=palette[i], lw=2)

    axes[1, 0].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    axes[1, 0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    axes[1, 0].set_title(f"3. One-vs-Rest ROC Curves (Macro AUC = {roc_auc_ovr:.3f})", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="lower right", fontsize=8.5)

    # Panel 4: Metrics KPI Summary Card
    axes[1, 1].axis("off")
    kpi_text = (
        "EVALUATION METRICS SUMMARY\n"
        "--------------------------------------------------\n"
        f"• Accuracy:               {acc*100:.2f}%\n"
        f"• Macro F1-Score:         {f1_macro*100:.2f}%\n"
        f"• Weighted F1-Score:      {f1_weighted*100:.2f}%\n"
        f"• Macro Precision:        {prec_macro*100:.2f}%\n"
        f"• Macro Recall:           {rec_macro*100:.2f}%\n"
        f"• Matthews Corr. (MCC):   {mcc:.4f}\n"
        f"• Cohen's Kappa (κ):      {kappa:.4f}\n"
        f"• Macro ROC-AUC:          {roc_auc_ovr:.4f}\n"
        f"• Cross-Entropy Loss:     {ce_loss:.4f}\n"
        "--------------------------------------------------\n"
        "Benchmark: Uncontrolled Potato Leaf Disease Dataset\n"
        f"Evaluated Test Specimens: {total_test}\n"
        f"Dual-Path Fusion Dim:     2,048 (1536 EffNet + 512 ViT)\n"
    )
    axes[1, 1].text(
        0.08, 0.5, kpi_text,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5)
    )
    axes[1, 1].set_title("4. Comprehensive Performance Metrics Card", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()
    print(f"[Artifact] Saved comprehensive 4-panel evaluation visual: {PLOT_PATH}", flush=True)


if __name__ == "__main__":
    run_comprehensive_evaluation()
