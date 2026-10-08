import sys
import os
import time
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

data = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff_feats = data['eff_feats'].float()
vit_cls = data['vit_cls'].float()
labels = data['labels'].long()

# Paper's exact split: 80% train, 10% val, 10% test (random_state=42)
idx_all = np.arange(len(labels))
idx_tr_val, idx_te = train_test_split(idx_all, test_size=0.10, random_state=42, stratify=labels.numpy())
idx_tr, idx_val = train_test_split(idx_tr_val, test_size=1/9, random_state=42, stratify=labels[idx_tr_val].numpy())

y_tr, y_val, y_te = labels[idx_tr].numpy(), labels[idx_val].numpy(), labels[idx_te].numpy()

# Normalize features
eff_norm = F.normalize(eff_feats, p=2, dim=1)
vit_norm = F.normalize(vit_cls, p=2, dim=1)

print("--- Testing Feature Normalization + Ridge / Logistic ---", flush=True)
for alpha in [0.1, 1.0, 5.0, 10.0, 20.0]:
    ridge = RidgeClassifier(alpha=alpha, random_state=42)
    X_fused_norm = torch.cat([eff_norm, vit_norm], dim=1).numpy()
    ridge.fit(X_fused_norm[idx_tr], y_tr)
    val_acc = accuracy_score(y_val, ridge.predict(X_fused_norm[idx_val]))
    te_acc = accuracy_score(y_te, ridge.predict(X_fused_norm[idx_te]))
    print(f"  Ridge alpha={alpha:<4} -> Val Acc: {val_acc*100:.2f}%, Test Acc: {te_acc*100:.2f}%", flush=True)

# Test ExtraTrees with normalized features
print("\n--- Testing ExtraTrees with Normalized Features ---", flush=True)
et = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)
et.fit(X_fused_norm[idx_tr], y_tr)
te_acc_et = accuracy_score(y_te, et.predict(X_fused_norm[idx_te]))
print(f"  ExtraTrees Test Acc: {te_acc_et*100:.2f}%", flush=True)

# Test Training on Train+Val (90% training data, 10% test data as per standard 90/10 protocol)
print("\n--- Testing Training on 90% (Train+Val) for Final Evaluation ---", flush=True)
et90 = ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1)
et90.fit(X_fused_norm[idx_tr_val], labels[idx_tr_val].numpy())
te_acc_et90 = accuracy_score(y_te, et90.predict(X_fused_norm[idx_te]))
print(f"  ExtraTrees 90% Train -> Test Acc: {te_acc_et90*100:.2f}%", flush=True)

# Test Deep Hybrid Head on L2-normalized features
print("\n--- Testing Deep Head on L2-normalized Features (70 epochs) ---", flush=True)
class NormalizedDeepHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.eff_proj = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1024 + 512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 7)
        )

    def forward(self, eff, vit):
        e = self.eff_proj(eff)
        v = self.vit_proj(vit)
        f = torch.cat([e, v], dim=1)
        return self.classifier(f)

torch.manual_seed(42)
m = NormalizedDeepHead()
opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70)
crit = nn.CrossEntropyLoss(label_smoothing=0.03)

ds = torch.utils.data.TensorDataset(eff_norm[idx_tr], vit_norm[idx_tr], labels[idx_tr])
loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

best_val_acc = 0.0
best_state = None
for ep in range(1, 71):
    m.train()
    for be, bv, by in loader:
        opt.zero_grad()
        loss = crit(m(be, bv), by)
        loss.backward()
        opt.step()
    sched.step()

    m.eval()
    with torch.no_grad():
        val_preds = m(eff_norm[idx_val], vit_norm[idx_val]).argmax(dim=1)
        vacc = (val_preds == labels[idx_val]).float().mean().item()
        if vacc > best_val_acc:
            best_val_acc = vacc
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

m.load_state_dict(best_state)
m.eval()
with torch.no_grad():
    tpreds = m(eff_norm[idx_te], vit_norm[idx_te]).argmax(dim=1).numpy()
    test_acc = accuracy_score(y_te, tpreds)
    print(f"  Normalized Deep Head -> Val Acc: {best_val_acc*100:.2f}%, Test Acc: {test_acc*100:.2f}%", flush=True)
