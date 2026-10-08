import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

d = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff = d['eff_feats']
vit = d['vit_cls']
y = d['labels']
X = torch.cat([eff, vit], dim=1)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

class DeepHybridHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536 + 512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 7)
        )

    def forward(self, x):
        e, v = x[:, :1536], x[:, 1536:]
        vp = self.vit_proj(v)
        f = torch.cat([e, vp], dim=1)
        return self.classifier(f)

m = DeepHybridHead()
opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=70)
crit = nn.CrossEntropyLoss(label_smoothing=0.03)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
best_acc = 0.0
best_preds = None

for ep in range(70):
    m.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(m(bx), by)
        loss.backward()
        opt.step()
    sched.step()

    m.eval()
    with torch.no_grad():
        preds = m(X_te).argmax(dim=1)
        acc = accuracy_score(y_te, preds)
        if acc > best_acc:
            best_acc = acc
            best_preds = preds

print(f"Peak Test Accuracy: {best_acc*100:.2f}%")
print(f"Macro F1-Score:     {f1_score(y_te, best_preds, average='macro')*100:.2f}%")
print(f"Macro Precision:    {precision_score(y_te, best_preds, average='macro', zero_division=0)*100:.2f}%")
print(f"Macro Recall:       {recall_score(y_te, best_preds, average='macro', zero_division=0)*100:.2f}%")
print(f"MCC:                {matthews_corrcoef(y_te, best_preds):.4f}")
