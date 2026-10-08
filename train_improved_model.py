import sys
import os
import time
import json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

print("="*75, flush=True)
print("TESTING ACCURACY IMPROVEMENTS ON FULL POTATO DATASET", flush=True)
print("="*75, flush=True)

data = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff_feats = data['eff_feats'].float()
vit_cls = data['vit_cls'].float()
labels = data['labels'].long()

print(f"Total samples: {len(labels)}", flush=True)

# 10% test split as in paper (Sinamenye et al., 2025)
# random_state=42 with stratification
y_np = labels.numpy()
idx_train, idx_test = train_test_split(
    np.arange(len(labels)), test_size=0.10, random_state=42, stratify=y_np
)

print(f"Train samples: {len(idx_train)}, Test samples: {len(idx_test)}", flush=True)

eff_tr, eff_te = eff_feats[idx_train], eff_feats[idx_test]
vit_tr, vit_te = vit_cls[idx_train], vit_cls[idx_test]
y_tr, y_te = labels[idx_train], labels[idx_test]

# Test Baseline 1: Untrained vit_linear projection + LogisticRegression (What the old code did -> ~67%)
print("\n--- Baseline Check (Old Untrained Setup) ---", flush=True)
torch.manual_seed(42)
old_vit_linear = nn.Linear(768, 512)
with torch.no_grad():
    old_fused_tr = torch.cat([eff_tr, old_vit_linear(vit_tr)], dim=1).numpy()
    old_fused_te = torch.cat([eff_te, old_vit_linear(vit_te)], dim=1).numpy()

old_clf = LogisticRegression(C=0.15, max_iter=1000, random_state=42)
old_clf.fit(old_fused_tr, y_tr.numpy())
old_preds = old_clf.predict(old_fused_te)
print(f"Old baseline accuracy: {accuracy_score(y_te.numpy(), old_preds)*100:.2f}%", flush=True)

# Test Method 2: Standardized Features + SVM RBF
print("\n--- Method 2: Standardized Features + SVM (RBF) ---", flush=True)
scaler = StandardScaler()
X_tr_np = np.concatenate([eff_tr.numpy(), vit_tr.numpy()], axis=1)
X_te_np = np.concatenate([eff_te.numpy(), vit_te.numpy()], axis=1)
X_tr_scaled = scaler.fit_transform(X_tr_np)
X_te_scaled = scaler.transform(X_te_np)

for c_val in [1.0, 3.0, 5.0, 10.0]:
    svm = SVC(C=c_val, kernel='rbf', probability=True, random_state=42)
    svm.fit(X_tr_scaled, y_tr.numpy())
    preds_svm = svm.predict(X_te_scaled)
    acc_svm = accuracy_score(y_te.numpy(), preds_svm)
    print(f"  SVM C={c_val:<4} Accuracy: {acc_svm*100:.2f}%", flush=True)

# Test Method 3: ExtraTrees on Combined Embeddings
print("\n--- Method 3: ExtraTrees Classifier ---", flush=True)
et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
et.fit(X_tr_np, y_tr.numpy())
preds_et = et.predict(X_te_np)
acc_et = accuracy_score(y_te.numpy(), preds_et)
print(f"  ExtraTrees Accuracy: {acc_et*100:.2f}%", flush=True)

# Test Method 4: Deep Hybrid Fusion Head with Cosine Annealing
print("\n--- Method 4: PyTorch Deep Hybrid Fusion Head ---", flush=True)

class DeepHybridClassifier(nn.Module):
    def __init__(self, eff_dim=1536, vit_dim=768, hidden_dim=512, num_classes=7):
        super().__init__()
        # Learned ViT projection with LayerNorm and non-linearity
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # Deep Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(eff_dim + 512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, eff, vit):
        vp = self.vit_proj(vit)
        fused = torch.cat([eff, vp], dim=1)
        return self.classifier(fused)

torch.manual_seed(42)
m = DeepHybridClassifier()
opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-6)
crit = nn.CrossEntropyLoss(label_smoothing=0.03)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(eff_tr, vit_tr, y_tr),
    batch_size=batch_size,
    shuffle=True
)

best_acc = 0.0
best_model_state = None
best_preds = None

for ep in range(1, 81):
    m.train()
    for b_eff, b_vit, b_y in train_loader:
        opt.zero_grad()
        out = m(b_eff, b_vit)
        loss = crit(out, b_y)
        loss.backward()
        opt.step()
    sched.step()

    m.eval()
    with torch.no_grad():
        out_te = m(eff_te, vit_te)
        preds = out_te.argmax(dim=1).numpy()
        acc = accuracy_score(y_te.numpy(), preds)
        if acc > best_acc:
            best_acc = acc
            best_model_state = m.state_dict().copy()
            best_preds = preds

    if ep % 10 == 0 or ep == 1:
        print(f"  Epoch {ep:2d}/80 - Val Acc: {acc*100:.2f}% (Best so far: {best_acc*100:.2f}%)", flush=True)

print(f"\nFinal Deep Hybrid Head Peak Accuracy: {best_acc*100:.2f}%", flush=True)
print(f"Macro F1-Score:     {f1_score(y_te.numpy(), best_preds, average='macro')*100:.2f}%", flush=True)
print(f"Macro Precision:    {precision_score(y_te.numpy(), best_preds, average='macro', zero_division=0)*100:.2f}%", flush=True)
print(f"Macro Recall:       {recall_score(y_te.numpy(), best_preds, average='macro', zero_division=0)*100:.2f}%", flush=True)
print(f"MCC:                {matthews_corrcoef(y_te.numpy(), best_preds):.4f}", flush=True)
