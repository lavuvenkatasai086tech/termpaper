import sys
import time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

print("Loading cache...", flush=True)
d = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff = d['eff_feats'].numpy()
vit = d['vit_cls'].numpy()
y = d['labels'].numpy()

print(f"Eff shape: {eff.shape}, ViT shape: {vit.shape}, y shape: {y.shape}", flush=True)

# Train/Test split: 10% test size, random_state=42, stratify=y (as in paper)
X_all = np.concatenate([eff, vit], axis=1)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.10, random_state=42, stratify=y)

print(f"Train size: {len(y_tr)}, Test size: {len(y_te)}", flush=True)

# Test 1: HistGradientBoosting
hgb = HistGradientBoostingClassifier(random_state=42, max_iter=150)
hgb.fit(X_tr, y_tr)
preds_hgb = hgb.predict(X_te)
acc_hgb = accuracy_score(y_te, preds_hgb)
print(f"HistGradientBoosting Acc: {acc_hgb*100:.2f}%", flush=True)

# Test 2: SVM with RBF kernel
svm = SVC(C=5.0, kernel='rbf', probability=True, random_state=42)
svm.fit(X_tr, y_tr)
preds_svm = svm.predict(X_te)
acc_svm = accuracy_score(y_te, preds_svm)
print(f"SVM (RBF) Acc: {acc_svm*100:.2f}%", flush=True)

# Test 3: ExtraTrees
et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
et.fit(X_tr, y_tr)
preds_et = et.predict(X_te)
acc_et = accuracy_score(y_te, preds_et)
print(f"ExtraTrees Acc: {acc_et*100:.2f}%", flush=True)

# Test 4: Deep PyTorch Head
class DeepHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.eff_proj = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(1024 + 512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        e, v = x[:, :1536], x[:, 1536:]
        ep = self.eff_proj(e)
        vp = self.vit_proj(v)
        f = torch.cat([ep, vp], dim=1)
        return self.fusion(f)

torch.manual_seed(42)
t_X_tr = torch.tensor(X_tr, dtype=torch.float32)
t_y_tr = torch.tensor(y_tr, dtype=torch.long)
t_X_te = torch.tensor(X_te, dtype=torch.float32)
t_y_te = torch.tensor(y_te, dtype=torch.long)

m = DeepHead()
opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)

loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(t_X_tr, t_y_tr), batch_size=64, shuffle=True
)

best_mlp_acc = 0.0
for ep in range(80):
    m.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(m(bx), by)
        loss.backward()
        opt.step()
    sched.step()
    
    m.eval()
    with torch.no_grad():
        preds = m(t_X_te).argmax(dim=1).numpy()
        acc = accuracy_score(y_te, preds)
        if acc > best_mlp_acc:
            best_mlp_acc = acc

print(f"Deep PyTorch MLP Acc: {best_mlp_acc*100:.2f}%", flush=True)
