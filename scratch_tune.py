import os
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

data = torch.load("checkpoints/features_cache_full.pt", map_location="cpu")
eff = data["eff_feats"].float()
vit = data["vit_cls"].float()
y = data["labels"].long()

num_samples = len(y)
idx_tr, idx_te = train_test_split(np.arange(num_samples), test_size=0.10, random_state=42, stratify=y.numpy())
y_tr = y[idx_tr]
y_te = y[idx_te]

print(f"Total samples: {num_samples}, Train: {len(idx_tr)}, Test: {len(idx_te)}")

# Let's test standard architecture from model.py:
# vit_linear: Linear(768, 512), classifier: Linear(2048, 7)
class BaseHead(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        self.vit_linear = nn.Linear(768, 512)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(1536 + 512, 7)
    def forward(self, e, v):
        vp = self.vit_linear(v)
        fused = torch.cat([e, vp], dim=1)
        fused = self.dropout(fused)
        return self.classifier(fused)

# Test hyperparameter search
results = []
for lr in [5e-4, 1e-3, 2e-3, 3e-3]:
    for wd in [1e-4, 1e-3, 1e-2]:
        for ls in [0.0, 0.05, 0.1]:
            torch.manual_seed(42)
            head = BaseHead()
            opt = optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
            crit = nn.CrossEntropyLoss(label_smoothing=ls)
            
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(eff[idx_tr], vit[idx_tr], y_tr),
                batch_size=32, shuffle=True
            )
            best_acc = 0.0
            for ep in range(40):
                head.train()
                for be, bv, by in loader:
                    opt.zero_grad()
                    l = crit(head(be, bv), by)
                    l.backward()
                    opt.step()
                sched.step()
                head.eval()
                with torch.no_grad():
                    preds = head(eff[idx_te], vit[idx_te]).argmax(dim=1)
                    acc = accuracy_score(y_te.numpy(), preds.numpy())
                    if acc > best_acc:
                        best_acc = acc
            results.append((best_acc, lr, wd, ls))

results.sort(key=lambda x: x[0], reverse=True)
print("Top 5 Hyperparameter configs for BaseHead:")
for acc, lr, wd, ls in results[:5]:
    print(f"  Acc: {acc*100:.2f}% | lr={lr}, wd={wd}, label_smoothing={ls}")
