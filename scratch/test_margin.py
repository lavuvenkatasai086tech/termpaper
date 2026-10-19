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

DATA_DIR = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\data\potatodata\Potato Leaf Disease Dataset in Uncontrolled Environment"
PV_DIR = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\PlantVillage"

def extract_f(img_p):
    img = Image.open(img_p).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        eff_f = model.effnet(t)
        v_out = model.vit(t)
        v_cls = v_out.last_hidden_state[:, 0, :]
        v_proj = model.vit_linear(v_cls)
        f = torch.cat([eff_f, v_proj], dim=1) # [1, 2048]
    return f.cpu()

# Collect 20 potato features and 20 non-potato features
potato_feats = []
from app import CURATED_CLASS_SAMPLES
for c, files in CURATED_CLASS_SAMPLES.items():
    for f in files:
        p = os.path.join(DATA_DIR, c, f)
        if os.path.exists(p):
            potato_feats.append(extract_f(p))

non_potato_feats = []
for folder in ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot", "Tomato_healthy", "Tomato_Early_blight", "Tomato_Leaf_Mold", "Tomato__Target_Spot"]:
    dir_p = os.path.join(PV_DIR, folder)
    if os.path.isdir(dir_p):
        for f in os.listdir(dir_p)[:4]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                non_potato_feats.append(extract_f(os.path.join(dir_p, f)))

P = torch.cat(potato_feats, dim=0) # [28, 2048]
NP = torch.cat(non_potato_feats, dim=0) # [24, 2048]

P_norm = F.normalize(P, p=2, dim=1)
NP_norm = F.normalize(NP, p=2, dim=1)

# Centroid of potato samples
p_centroid = F.normalize(P_norm.mean(dim=0, keepdim=True), p=2, dim=1)
# Centroid of non-potato samples
np_centroid = F.normalize(NP_norm.mean(dim=0, keepdim=True), p=2, dim=1)

p_sim_to_p = torch.matmul(P_norm, p_centroid.t()).squeeze()
p_sim_to_np = torch.matmul(P_norm, np_centroid.t()).squeeze()

np_sim_to_p = torch.matmul(NP_norm, p_centroid.t()).squeeze()
np_sim_to_np = torch.matmul(NP_norm, np_centroid.t()).squeeze()

print("POTATO SAMPLES vs Centroids:")
print(f"  Sim to Potato Centroid:     min={p_sim_to_p.min():.4f}, mean={p_sim_to_p.mean():.4f}, max={p_sim_to_p.max():.4f}")
print(f"  Sim to Non-Potato Centroid: min={p_sim_to_np.min():.4f}, mean={p_sim_to_np.mean():.4f}, max={p_sim_to_np.max():.4f}")
print(f"  Potato Margin (Sim_P - Sim_NP): min={(p_sim_to_p - p_sim_to_np).min():.4f}, mean={(p_sim_to_p - p_sim_to_np).mean():.4f}")

print("\nNON-POTATO SAMPLES vs Centroids:")
print(f"  Sim to Potato Centroid:     min={np_sim_to_p.min():.4f}, mean={np_sim_to_p.mean():.4f}, max={np_sim_to_p.max():.4f}")
print(f"  Sim to Non-Potato Centroid: min={np_sim_to_np.min():.4f}, mean={np_sim_to_np.mean():.4f}, max={np_sim_to_np.max():.4f}")
print(f"  Non-Potato Margin (Sim_P - Sim_NP): min={(np_sim_to_p - np_sim_to_np).min():.4f}, max={(np_sim_to_p - np_sim_to_np).max():.4f}")
