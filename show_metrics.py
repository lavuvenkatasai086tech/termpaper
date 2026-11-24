"""
Quick Metrics Display Script for VS Code Terminal
Reads evaluation_report.json and prints a formatted terminal summary.
"""
import os
import json

REPORT_PATH = os.path.join(os.path.dirname(__file__), "evaluation_report.json")

def display_metrics():
    if not os.path.exists(REPORT_PATH):
        print(f"Error: {REPORT_PATH} not found. Run 'python run_evaluation.py' first.")
        return

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = data.get("overall_metrics", {})
    per_class = data.get("per_class_metrics", [])

    print("\n" + "=" * 70)
    print("🌿 POTATO LEAF DISEASE DETECTION — MODEL EVALUATION BENCHMARKS")
    print("   Architecture: Enhanced Hybrid EfficientNetV2-B3 + ViT (Sinamenye et al., 2025)")
    print(f"   Evaluated Test Specimens: {data.get('test_samples_count', 308)}")
    print("=" * 70)

    print("\n📈 1. OVERALL TEST SET METRICS:")
    print("-" * 50)
    print(f"  • Test Accuracy              : {m.get('accuracy', 0)*100:6.2f}%")
    print(f"  • Macro F1-Score             : {m.get('macro_f1', 0)*100:6.2f}%")
    print(f"  • Weighted F1-Score          : {m.get('weighted_f1', 0)*100:6.2f}%")
    print(f"  • Macro Precision            : {m.get('macro_precision', 0)*100:6.2f}%")
    print(f"  • Macro Recall (Sensitivity) : {m.get('macro_recall', 0)*100:6.2f}%")
    print(f"  • Matthews Corr. Coeff (MCC) : {m.get('mcc', 0):6.4f}")
    print(f"  • Cohen's Kappa (κ)          : {m.get('cohens_kappa', 0):6.4f}")
    print(f"  • Macro ROC-AUC (OVR)        : {m.get('roc_auc_macro_ovr', 0):6.4f}")
    print(f"  • Cross-Entropy Loss         : {m.get('cross_entropy_loss', 0):6.4f}")
    print("-" * 50)

    print("\n📊 2. COMPARISON WITH PUBLISHED PAPER (Sinamenye et al., 2025):")
    print("-" * 70)
    print(f"  {'Metric':<24s} | {'Paper Baseline':<16s} | {'Your Trained Model':<16s} | {'Delta':<8s}")
    print("-" * 70)
    paper = {"Accuracy": 85.06, "F1-Score": 83.77, "Precision": 82.86, "MCC": 0.8200}
    ours = {
        "Accuracy": m.get("accuracy", 0)*100,
        "F1-Score": m.get("macro_f1", 0)*100,
        "Precision": m.get("macro_precision", 0)*100,
        "MCC": m.get("mcc", 0),
    }
    for k in paper:
        p_val = paper[k]
        o_val = ours[k]
        diff = o_val - p_val
        unit = "%" if k != "MCC" else ""
        diff_str = f"+{diff:.2f}{unit}" if diff > 0 else f"{diff:.2f}{unit}"
        print(f"  {k:<24s} | {p_val:6.2f}{unit:<10s} | {o_val:6.2f}{unit:<10s} | {diff_str:<8s}")
    print("-" * 70)

    print("\n🔬 3. PER-CLASS PATHOLOGY PERFORMANCE BREAKDOWN:")
    print("-" * 75)
    print(f"  {'Class':<14s} | {'Support':<8s} | {'Precision':<10s} | {'Recall':<10s} | {'F1-Score':<10s} | {'ROC-AUC':<8s}")
    print("-" * 75)
    for c in per_class:
        cls_name = c.get("class", "N/A")
        supp = c.get("support", 0)
        prec = c.get("precision", 0) * 100
        rec = c.get("recall_sensitivity", 0) * 100
        f1 = c.get("f1_score", 0) * 100
        auc = c.get("roc_auc", 0)
        print(f"  {cls_name:<14s} | {supp:<8d} | {prec:8.2f}%  | {rec:8.2f}%  | {f1:8.2f}%  | {auc:7.3f}")
    print("-" * 75)
    print("\n💡 Visual Artifacts Available:")
    print("  • 4-Panel Analysis Plot : code evaluation_metrics.png")
    print("  • Confusion Matrix Plot : code confusion_matrix.png")
    print("  • Raw JSON Report       : code evaluation_report.json\n")

if __name__ == "__main__":
    display_metrics()
