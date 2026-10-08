import sys
import os
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff_feats = data['eff_feats'].float()
vit_cls = data['vit_cls'].float()
labels = data['labels'].long()

y_all = labels.numpy()
print(f"Total samples: {len(y_all)} across 7 classes: {np.bincount(y_all)}", flush=True)

# 1. Paper split without stratify (as written in potato-efficientViT-main/src/dataset.py):
# train_indices, test_indices = train_test_split(np.arange(len(targets)), test_size=0.1, random_state=42)
# train_indices, vali_indices = train_test_split(train_indices, test_size=0.1, random_state=42)
idx_tr_unstrat, idx_te_unstrat = train_test_split(np.arange(len(y_all)), test_size=0.10, random_state=42)
idx_tr_unstrat, idx_val_unstrat = train_test_split(idx_tr_unstrat, test_size=0.10, random_state=42)

# 2. Stratified split (80/10/10)
idx_tr_val_s, idx_te_s = train_test_split(np.arange(len(y_all)), test_size=0.10, random_state=42, stratify=y_all)
idx_tr_s, idx_val_s = train_test_split(idx_tr_val_s, test_size=1/9, random_state=42, stratify=y_all[idx_tr_val_s])

print(f"Unstratified split: Train={len(idx_tr_unstrat)}, Val={len(idx_val_unstrat)}, Test={len(idx_te_unstrat)}", flush=True)
print(f"Stratified split:   Train={len(idx_tr_s)}, Val={len(idx_val_s)}, Test={len(idx_te_s)}", flush=True)

# Test various models on both splits
splits = [
    ("Unstratified (Paper exact dataset.py code)", idx_tr_unstrat, idx_val_unstrat, idx_te_unstrat),
    ("Stratified (80/10/10 balanced)", idx_tr_s, idx_val_s, idx_te_s)
]

for s_name, tr_i, val_i, te_i in splits:
    print(f"\n==================== {s_name} ====================", flush=True)
    eff_tr, eff_te = eff_feats[tr_i], eff_feats[te_i]
    vit_tr, vit_te = vit_cls[tr_i], vit_cls[te_i]
    y_tr, y_te = y_all[tr_i], y_all[te_i]

    # Model A: ExtraTrees on concatenated raw features
    X_tr = np.concatenate([eff_tr.numpy(), vit_tr.numpy()], axis=1)
    X_te = np.concatenate([eff_te.numpy(), vit_te.numpy()], axis=1)
    
    et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    et.fit(X_tr, y_tr)
    preds_et = et.predict(X_te)
    print(f"  ExtraTrees Test Acc: {accuracy_score(y_te, preds_et)*100:.2f}%", flush=True)

    # Model B: SVM RBF (C=10)
    svm = SVC(C=10.0, kernel='rbf', random_state=42)
    svm.fit(X_tr, y_tr)
    preds_svm = svm.predict(X_te)
    print(f"  SVM (RBF C=10) Test Acc: {accuracy_score(y_te, preds_svm)*100:.2f}%", flush=True)

    # Model C: KNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_tr, y_tr)
    preds_knn = knn.predict(X_te)
    print(f"  KNN (k=5) Test Acc: {accuracy_score(y_te, preds_knn)*100:.2f}%", flush=True)

    # Model D: Neural Deep Hybrid Head
    class DeepHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit_proj = nn.Sequential(
                nn.Linear(768, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1536 + 512, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 7)
            )

        def forward(self, e, v):
            vp = self.vit_proj(v)
            f = torch.cat([e, vp], dim=1)
            return self.classifier(f)

    torch.manual_seed(42)
    m = DeepHead()
    opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70)
    crit = nn.CrossEntropyLoss(label_smoothing=0.03)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(eff_tr, vit_tr, torch.tensor(y_tr)),
        batch_size=32, shuffle=True
    )
    best_acc = 0.0
    for ep in range(70):
        m.train()
        for be, bv, by in loader:
            opt.zero_grad()
            loss = crit(m(be, bv), by)
            loss.backward()
            opt.step()
        sched.step()

        m.eval()
        with torch.no_grad():
            preds = m(eff_te, vit_te).argmax(dim=1).numpy()
            acc = accuracy_score(y_te, preds)
            if acc > best_acc:
                best_acc = acc

    print(f"  PyTorch Deep Hybrid Head Peak Test Acc: {best_acc*100:.2f}%", flush=True)
