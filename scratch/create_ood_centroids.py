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
        f = torch.cat([eff_f, v_proj], dim=1)
    return f.cpu()

print("1. Extracting potato features...", flush=True)
from app import CURATED_CLASS_SAMPLES
potato_feats = []
for c, files in CURATED_CLASS_SAMPLES.items():
    for f in files:
        p = os.path.join(DATA_DIR, c, f)
        if os.path.exists(p):
            potato_feats.append(extract_f(p))

print(f"   Collected {len(potato_feats)} potato samples.", flush=True)

print("2. Extracting non-potato features...", flush=True)
non_potato_feats = []
for folder in ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot", "Tomato_healthy", "Tomato_Early_blight", "Tomato_Leaf_Mold", "Tomato__Target_Spot"]:
    dir_p = os.path.join(PV_DIR, folder)
    if os.path.isdir(dir_p):
        for f in os.listdir(dir_p)[:4]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                non_potato_feats.append(extract_f(os.path.join(dir_p, f)))

print(f"   Collected {len(non_potato_feats)} non-potato samples.", flush=True)

P = torch.cat(potato_feats, dim=0) # [28, 2048]
NP = torch.cat(non_potato_feats, dim=0) # [24, 2048]

P_norm = F.normalize(P, p=2, dim=1)
NP_norm = F.normalize(NP, p=2, dim=1)

p_centroid = F.normalize(P_norm.mean(dim=0, keepdim=True), p=2, dim=1)
np_centroid = F.normalize(NP_norm.mean(dim=0, keepdim=True), p=2, dim=1)

p_sim_to_p = torch.matmul(P_norm, p_centroid.t()).squeeze()
p_sim_to_np = torch.matmul(P_norm, np_centroid.t()).squeeze()

np_sim_to_p = torch.matmul(NP_norm, p_centroid.t()).squeeze()
np_sim_to_np = torch.matmul(NP_norm, np_centroid.t()).squeeze()

print(f"\nPOTATO min Sim_P: {p_sim_to_p.min():.4f}, min Margin (P - NP): {(p_sim_to_p - p_sim_to_np).min():.4f}", flush=True)
print(f"NON-POTATO max Sim_P: {np_sim_to_p.max():.4f}, max Margin (P - NP): {(np_sim_to_p - np_sim_to_np).max():.4f}", flush=True)

save_data = {
    'potato_centroid': p_centroid.squeeze(0),         # [2048]
    'non_potato_centroid': np_centroid.squeeze(0),     # [2048]
    'sim_threshold': 0.49,
    'margin_threshold': 0.05
}
torch.save(save_data, 'checkpoints/potato_ood_centroids.pt')
print("Saved checkpoints/potato_ood_centroids.pt successfully!", flush=True)
