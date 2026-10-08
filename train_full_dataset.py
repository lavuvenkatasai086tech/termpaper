import os
import sys
import time
import json
import random

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score, roc_curve,
    log_loss, confusion_matrix, classification_report
)
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
FULL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "features_cache_full.pt")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
PLOT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_metrics.png")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_all_samples():
    """
    Scans the entire dataset directory and collects all available images across all 7 classes.
    """
    samples = []
    print(f"[Dataset] Scanning entire dataset from: {DATA_DIR}", flush=True)
    class_counts = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_folder):
            continue
        all_files = [
            os.path.join(class_folder, f)
            for f in os.listdir(class_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        class_counts[class_name] = len(all_files)
        print(f"  - {class_name:12s}: {len(all_files):4d} images", flush=True)
        for img_path in all_files:
            samples.append((img_path, class_idx))

    random.shuffle(samples)
    print(f"[Dataset] Total images loaded across 100% of dataset: {len(samples)}", flush=True)
    return samples, class_counts


def extract_full_features(model, samples, batch_size=32, device="cpu"):
    """
    Extracts frozen EfficientNetV2B3 and ViT features across all 3,076 images.
    """
    if os.path.exists(FULL_CACHE_PATH):
        print(f"[Cache] Found pre-extracted full dataset features at: {FULL_CACHE_PATH}", flush=True)
        data = torch.load(FULL_CACHE_PATH, map_location="cpu")
        return data["eff_feats"], data["vit_cls"], data["labels"]

    print(f"[Extraction] Extracting features for all {len(samples)} images (batch size {batch_size}) on {device}...", flush=True)
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
                print(f"  Warning: failed to read {img_path}: {e}", flush=True)

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
        eta_sec = (len(samples) - done_imgs) / max(rate, 0.001)
        eta_min = eta_sec / 60.0

        if (b_idx + 1) % 5 == 0 or (b_idx + 1) == total_batches or (b_idx + 1) <= 3:
            print(f"  Batch {b_idx + 1:2d}/{total_batches} [{done_imgs:4d}/{len(samples)}] - {rate:.2f} img/s - ETA: {eta_min:.1f} min ({eta_sec:.0f}s)", flush=True)

    eff_feats_tensor = torch.cat(all_eff_feats, dim=0)
    vit_cls_tensor = torch.cat(all_vit_cls, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    os.makedirs(os.path.dirname(FULL_CACHE_PATH), exist_ok=True)
    torch.save({
        "eff_feats": eff_feats_tensor,
        "vit_cls": vit_cls_tensor,
        "labels": labels_tensor
    }, FULL_CACHE_PATH)
    print(f"[Cache] Successfully cached all {len(labels_tensor)} features to: {FULL_CACHE_PATH}", flush=True)

    return eff_feats_tensor, vit_cls_tensor, labels_tensor


def main():
    set_seed(42)
    device = torch.device("cpu")
    print("=" * 80, flush=True)
    print("[Full Dataset Training Pipeline] EfficientNetV2B3 + Vision Transformer", flush=True)
    print("   Dataset: Potato Leaf Disease Dataset in Uncontrolled Environment", flush=True)
    print(f"   Device:  {device}", flush=True)
    print("=" * 80, flush=True)

    # 1. Initialize Hybrid Model
    print("\n[Step 1/5] Initializing Hybrid Architecture...", flush=True)
    model = EfficientNetV2B3ViT(num_classes=len(CLASS_NAMES))
    model.to(device)
    model.eval()

    # 2. Collect All Samples
    print("\n[Step 2/5] Collecting all images...", flush=True)
    samples, class_counts = collect_all_samples()

    # 3. Extract or Load Precomputed Features
    print("\n[Step 3/5] Extracting deep representations for 100% of dataset...", flush=True)
    eff_feats, vit_cls, labels = extract_full_features(model, samples, batch_size=32, device=device)
    y_all = labels.numpy()

    with torch.no_grad():
        vit_proj = model.vit_linear(vit_cls)
        fused = torch.cat([eff_feats, vit_proj], dim=1).numpy()

    print(f"Fused Feature Matrix: {fused.shape[0]} samples x {fused.shape[1]} dimensions", flush=True)

    # 4. Stratified Cross-Validation & Holdout Evaluation
    print("\n[Step 4/5] Evaluating Stratified 5-Fold Cross Validation...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs, cv_f1s = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(fused, y_all)):
        clf = LogisticRegression(C=0.15, max_iter=1500, class_weight="balanced", random_state=42)
        clf.fit(fused[tr_idx], y_all[tr_idx])
        preds = clf.predict(fused[val_idx])
        cv_accs.append(accuracy_score(y_all[val_idx], preds))
        cv_f1s.append(f1_score(y_all[val_idx], preds, average="macro", zero_division=0))

    print(f"  5-Fold CV Accuracy: {np.mean(cv_accs)*100:.2f}% (+/- {np.std(cv_accs)*100:.2f}%)", flush=True)
    print(f"  5-Fold CV Macro F1: {np.mean(cv_f1s)*100:.2f}% (+/- {np.std(cv_f1s)*100:.2f}%)", flush=True)

    # Holdout Test Split (15% unseen holdout)
    X_train, X_test, y_train, y_test = train_test_split(
        fused, y_all, test_size=0.15, random_state=42, stratify=y_all
    )

    eval_clf = LogisticRegression(C=0.15, max_iter=1500, class_weight="balanced", random_state=42)
    eval_clf.fit(X_train, y_train)

    test_preds = eval_clf.predict(X_test)
    test_probs = eval_clf.predict_proba(X_test)

    acc = accuracy_score(y_test, test_preds)
    prec_macro = precision_score(y_test, test_preds, average="macro", zero_division=0)
    prec_weighted = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    rec_macro = recall_score(y_test, test_preds, average="macro", zero_division=0)
    rec_weighted = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, test_preds, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, test_preds)
    kappa = cohen_kappa_score(y_test, test_preds)
    ce_loss = log_loss(y_test, test_probs)

    y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_NAMES))))
    roc_auc_ovr = roc_auc_score(y_test_bin, test_probs, average="macro", multi_class="ovr")
    roc_auc_weighted = roc_auc_score(y_test_bin, test_probs, average="weighted", multi_class="ovr")

    print("\n" + "=" * 55, flush=True)
    print("[FINAL TEST BENCHMARK METRICS (100% DATASET POOL)]", flush=True)
    print("=" * 55, flush=True)
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
    print("=" * 55, flush=True)

    cm = confusion_matrix(y_test, test_preds)
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
            auc_val = roc_auc_score(y_test_bin[:, i], test_probs[:, i])
        except Exception:
            auc_val = 0.5

        per_class_stats.append({
            "class": c_name,
            "support": int(cm[i, :].sum()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": round(float(p_val), 4),
            "recall_sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "f1_score": round(float(f1_val), 4),
            "roc_auc": round(float(auc_val), 4),
        })

        print(f"{c_name:14s} | {cm[i, :].sum():7d} | {p_val*100:5.1f}% | {sens*100:7.1f}% | {spec*100:5.1f}% | {f1_val*100:7.1f}% | {auc_val:5.3f}", flush=True)

    print("-" * 75, flush=True)

    # 5. Fit Deployment Classifier on Full Dataset & Update Checkpoint
    print("\n[Step 5/5] Fitting deployment head and saving weights...", flush=True)
    deploy_clf = LogisticRegression(C=0.15, max_iter=2000, class_weight="balanced", random_state=42)
    deploy_clf.fit(fused, y_all)

    model.classifier.weight.data = torch.tensor(deploy_clf.coef_, dtype=torch.float32)
    model.classifier.bias.data = torch.tensor(deploy_clf.intercept_, dtype=torch.float32)

    with torch.no_grad():
        pt_logits = model.classifier(torch.tensor(fused, dtype=torch.float32))
        pt_preds = pt_logits.argmax(dim=1).numpy()
        assert np.array_equal(pt_preds, deploy_clf.predict(fused)), "PyTorch and Sklearn prediction mismatch!"
        print("  [Verification] Weights verified with 100% precision match.", flush=True)

    # Save Checkpoint
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "metrics": {
            "test_accuracy": acc,
            "test_precision": prec_macro,
            "test_recall": rec_macro,
            "test_f1": f1_macro,
            "test_mcc": mcc,
            "cv_accuracy_mean": float(np.mean(cv_accs)),
            "total_dataset_size": len(y_all),
        },
        "model_type": "EfficientNetV2B3ViT",
        "timestamp": time.time(),
    }
    torch.save(checkpoint_payload, CHECKPOINT_PATH)
    print(f"  [Checkpoint] Saved best_model.pth ({os.path.getsize(CHECKPOINT_PATH):,} bytes)", flush=True)

    # Save JSON Report
    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "EfficientNetV2B3 + ViT (Sinamenye et al., 2025)",
        "total_dataset_images": len(y_all),
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
            "cv_accuracy_mean": round(float(np.mean(cv_accs)), 4),
        },
        "per_class_metrics": per_class_stats,
        "confusion_matrix": cm.tolist(),
        "classes": CLASS_NAMES,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  [Artifact] Saved metrics report JSON: {REPORT_PATH}", flush=True)

    # Save 4-Panel Visualization
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Panel 1: Normalized Confusion Matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt=".1%", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=axes[0, 0], cbar=True
    )
    axes[0, 0].set_title("1. Normalized Confusion Matrix (Sensitivity per Class)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Predicted Class", fontsize=10)
    axes[0, 0].set_ylabel("True Class", fontsize=10)
    axes[0, 0].tick_params(axis="x", rotation=35)

    # Panel 2: Precision vs Recall by Pathology
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
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
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
        "EVALUATION METRICS SUMMARY (100% DATASET)\n"
        "--------------------------------------------------\n"
        f"• Total Dataset Images:   {len(y_all)}\n"
        f"• Test Accuracy:          {acc*100:.2f}%\n"
        f"• 5-Fold CV Accuracy:     {np.mean(cv_accs)*100:.2f}%\n"
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
        f"Evaluated Test Holdout:   {total_test} images\n"
        f"Dual-Path Fusion Dim:     2,048 (1536 EffNet + 512 ViT)\n"
    )
    axes[1, 1].text(
        0.05, 0.5, kpi_text,
        fontsize=10.5,
        fontfamily="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=1", facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5)
    )
    axes[1, 1].set_title("4. Comprehensive Performance Metrics Card", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()
    print(f"  [Artifact] Saved 4-panel visual: {PLOT_PATH}", flush=True)
    print("\n[Complete] Pipeline finished successfully!", flush=True)


if __name__ == "__main__":
    main()
