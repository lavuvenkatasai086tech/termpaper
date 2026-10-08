import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier

data = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff_feats = data['eff_feats'].float()
vit_cls = data['vit_cls'].float()
labels = data['labels'].long()
y_all = labels.numpy()

idx_train, idx_test = train_test_split(np.arange(len(y_all)), test_size=0.10, random_state=42, stratify=y_all)

# Model 1: Checkpoint loaded from best_model.pth
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
from model import EfficientNetV2B3ViT
model = EfficientNetV2B3ViT(num_classes=7)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

with torch.no_grad():
    vp = model.vit_linear(vit_cls[idx_test])
    fused_te = torch.cat([eff_feats[idx_test], vp], dim=1)
    probs_pt = F.softmax(model.classifier(fused_te), dim=1).numpy()

# Model 2: Ridge classifier on L2-normalized features
eff_norm = F.normalize(eff_feats, p=2, dim=1)
vit_norm = F.normalize(vit_cls, p=2, dim=1)
fused_norm = torch.cat([eff_norm, vit_norm], dim=1).numpy()

lr = LogisticRegression(C=2.0, max_iter=1000, random_state=42)
lr.fit(fused_norm[idx_train], y_all[idx_train])
probs_lr = lr.predict_proba(fused_norm[idx_test])

# Model 3: ExtraTrees
et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
et.fit(fused_norm[idx_train], y_all[idx_train])
probs_et = et.predict_proba(fused_norm[idx_test])

print("Individual Accuracies:")
print(f"  PyTorch Model:    {accuracy_score(y_all[idx_test], probs_pt.argmax(axis=1))*100:.2f}%")
print(f"  LogisticReg (L2): {accuracy_score(y_all[idx_test], probs_lr.argmax(axis=1))*100:.2f}%")
print(f"  ExtraTrees:       {accuracy_score(y_all[idx_test], probs_et.argmax(axis=1))*100:.2f}%")

# Blended Ensemble:
for w_pt in [0.5, 0.6, 0.7, 0.8]:
    w_lr = (1.0 - w_pt) / 2
    w_et = (1.0 - w_pt) / 2
    blend_probs = w_pt * probs_pt + w_lr * probs_lr + w_et * probs_et
    blend_preds = blend_probs.argmax(axis=1)
    acc_blend = accuracy_score(y_all[idx_test], blend_preds)
    print(f"  Ensemble w_pt={w_pt} -> Test Acc: {acc_blend*100:.2f}%, F1: {f1_score(y_all[idx_test], blend_preds, average='macro')*100:.2f}%")
