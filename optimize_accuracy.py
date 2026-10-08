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

data = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff_feats = data['eff_feats'].float()
vit_cls = data['vit_cls'].float()
labels = data['labels'].long()

print(f"Total dataset: {len(labels)} images across 7 classes", flush=True)

# Paper split: 80% train, 10% val, 10% test (random_state=42, stratified)
# First split 90% train_val, 10% test
idx_all = np.arange(len(labels))
idx_tr_val, idx_te = train_test_split(idx_all, test_size=0.10, random_state=42, stratify=labels.numpy())
# Then split 90% into 80% train and 10% val (1/9 of 90% is 10%)
idx_tr, idx_val = train_test_split(idx_tr_val, test_size=1/9, random_state=42, stratify=labels[idx_tr_val].numpy())

print(f"Split sizes -> Train: {len(idx_tr)} ({len(idx_tr)/len(labels)*100:.1f}%), Val: {len(idx_val)} ({len(idx_val)/len(labels)*100:.1f}%), Test: {len(idx_te)} ({len(idx_te)/len(labels)*100:.1f}%)", flush=True)

# Architecture 1: Standard Paper Head (vit_linear 768->512, Dropout(0.2), Linear 2048->7)
class StandardPaperHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_linear = nn.Linear(768, 512)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1536 + 512, 7)

    def forward(self, eff, vit):
        vp = self.vit_linear(vit)
        fused = torch.cat([eff, vp], dim=1)
        return self.classifier(self.dropout(fused))

# Architecture 2: Deep Non-Linear Hybrid Head
class DeepHybridHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_linear = nn.Sequential(
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

    def forward(self, eff, vit):
        vp = self.vit_linear(vit)
        fused = torch.cat([eff, vp], dim=1)
        return self.classifier(fused)

# Architecture 3: Attention-Gated Multi-Head Fusion
class AttentiveHybridHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_linear = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU()
        )
        # Cross-gating between CNN texture and ViT context
        self.gate = nn.Sequential(
            nn.Linear(1536 + 512, 512),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536 + 512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 7)
        )

    def forward(self, eff, vit):
        vp = self.vit_linear(vit)
        fused = torch.cat([eff, vp], dim=1)
        return self.classifier(fused)

def train_and_eval(head_class, name, lr=1e-3, epochs=70, batch_size=32, weight_decay=1e-4, label_smoothing=0.0):
    torch.manual_seed(42)
    np.random.seed(42)
    
    m = head_class()
    opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    ds_tr = torch.utils.data.TensorDataset(eff_feats[idx_tr], vit_cls[idx_tr], labels[idx_tr])
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None

    t0 = time.time()
    for ep in range(1, epochs + 1):
        m.train()
        for b_eff, b_vit, b_y in loader:
            opt.zero_grad()
            out = m(b_eff, b_vit)
            loss = crit(out, b_y)
            loss.backward()
            opt.step()
        sched.step()

        m.eval()
        with torch.no_grad():
            val_out = m(eff_feats[idx_val], vit_cls[idx_val])
            val_preds = val_out.argmax(dim=1)
            val_acc = (val_preds == labels[idx_val]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

    # Evaluate best validation checkpoint on unseen TEST split
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        test_out = m(eff_feats[idx_te], vit_cls[idx_te])
        test_preds = test_out.argmax(dim=1).numpy()
        test_y = labels[idx_te].numpy()

    test_acc = accuracy_score(test_y, test_preds)
    test_f1 = f1_score(test_y, test_preds, average='macro')
    test_prec = precision_score(test_y, test_preds, average='macro', zero_division=0)
    test_rec = recall_score(test_y, test_preds, average='macro', zero_division=0)
    test_mcc = matthews_corrcoef(test_y, test_preds)

    print(f"\n[{name}] Completed in {time.time()-t0:.1f}s", flush=True)
    print(f"  Best Val Accuracy:  {best_val_acc*100:.2f}%", flush=True)
    print(f"  Test Accuracy:      {test_acc*100:.2f}%", flush=True)
    print(f"  Test Macro F1:      {test_f1*100:.2f}%", flush=True)
    print(f"  Test Precision:     {test_prec*100:.2f}%", flush=True)
    print(f"  Test Recall:        {test_rec*100:.2f}%", flush=True)
    print(f"  Test MCC:           {test_mcc:.4f}", flush=True)
    return test_acc, m, best_state

print("\n--- Testing Architecture 1: Standard Paper Head ---", flush=True)
train_and_eval(StandardPaperHead, "Standard Paper Head (vit_linear + Linear)", lr=1e-3, epochs=70)

print("\n--- Testing Architecture 2: Deep Hybrid Head ---", flush=True)
train_and_eval(DeepHybridHead, "Deep Hybrid Head (LayerNorm + GELU + 2-layer MLP)", lr=1e-3, epochs=70)

print("\n--- Testing Architecture 3: Attentive Hybrid Head ---", flush=True)
train_and_eval(AttentiveHybridHead, "Attentive Hybrid Head", lr=1e-3, epochs=70)
