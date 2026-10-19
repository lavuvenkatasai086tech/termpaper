import os
import sys
sys.path.insert(0, r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project")
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from dataset import get_transforms
from model import EfficientNetV2B3ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt = torch.load('checkpoints/best_model.pth', map_location=device)
sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
model = EfficientNetV2B3ViT(num_classes=7).to(device)
model.load_state_dict(sd)
model.eval()

transform = get_transforms(augment=False)

# Load cached training features
cache = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff_train = cache['eff_feats']  # [3076, 1536]
vit_cls_train = cache['vit_cls'] # [3076, 768]
labels_train = cache['labels']  # [3076]

vit_w = sd['vit_linear.weight']
vit_b = sd['vit_linear.bias']
vit_proj_train = torch.matmul(vit_cls_train, vit_w.t()) + vit_b
fused_train = torch.cat([eff_train, vit_proj_train], dim=1) # [3076, 2048]

# Normalize features
fused_train_norm = F.normalize(fused_train, p=2, dim=1)

# Compute mean feature vector for each of the 7 classes
class_prototypes = []
for c in range(7):
    mask = (labels_train == c)
    proto = fused_train_norm[mask].mean(dim=0)
    proto = F.normalize(proto, p=2, dim=0)
    class_prototypes.append(proto)
class_prototypes = torch.stack(class_prototypes, dim=0) # [7, 2048]

global_potato_prototype = F.normalize(fused_train_norm.mean(dim=0), p=2, dim=0) # [2048]

def get_feat(path):
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        eff_f = model.effnet(t).cpu()
        vit_out = model.vit(t)
        v_cls = vit_out.last_hidden_state[:, 0, :].cpu()
        v_proj = torch.matmul(v_cls, vit_w.t()) + vit_b
        f = torch.cat([eff_f, v_proj], dim=1)
        f_norm = F.normalize(f, p=2, dim=1).squeeze(0)
    return f_norm

print("--- Testing Potato Images ---", flush=True)
DATA_DIR = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\data\potatodata\Potato Leaf Disease Dataset in Uncontrolled Environment"
from app import CURATED_CLASS_SAMPLES
for c in ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]:
    f = CURATED_CLASS_SAMPLES[c][0]
    p = os.path.join(DATA_DIR, c, f)
    fn = get_feat(p)
    cos_global = torch.dot(fn, global_potato_prototype).item()
    cos_max_class = torch.matmul(class_prototypes, fn).max().item()
    # Also kNN cosine similarity (average of top-5 nearest neighbors in fused_train_norm)
    top_knn_sim = torch.matmul(fused_train_norm, fn).topk(5).values.mean().item()
    print(f"POTATO {c:12s} | Global: {cos_global:.4f} | MaxClass: {cos_max_class:.4f} | Top5-kNN: {top_knn_sim:.4f}", flush=True)

print("\n--- Testing Non-Potato PlantVillage (Pepper & Tomato) ---", flush=True)
pv_root = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\PlantVillage"
for folder in ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot", "Tomato_healthy", "Tomato_Early_blight"]:
    f_path = os.path.join(pv_root, folder)
    if os.path.isdir(f_path):
        imgs = [os.path.join(f_path, x) for x in os.listdir(f_path) if x.endswith('.JPG') or x.endswith('.jpg')][:2]
        for idx, img_p in enumerate(imgs):
            fn = get_feat(img_p)
            cos_global = torch.dot(fn, global_potato_prototype).item()
            cos_max_class = torch.matmul(class_prototypes, fn).max().item()
            top_knn_sim = torch.matmul(fused_train_norm, fn).topk(5).values.mean().item()
            print(f"NON-POTATO {folder[:20]:20s} #{idx+1} | Global: {cos_global:.4f} | MaxClass: {cos_max_class:.4f} | Top5-kNN: {top_knn_sim:.4f}", flush=True)
