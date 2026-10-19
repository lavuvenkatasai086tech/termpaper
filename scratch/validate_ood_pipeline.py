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

def extract_f(img):
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        eff_f = model.effnet(t)
        v_out = model.vit(t)
        v_cls = v_out.last_hidden_state[:, 0, :]
        v_proj = model.vit_linear(v_cls)
        f = torch.cat([eff_f, v_proj], dim=1) # [1, 2048]
    return f.cpu()

# Load saved centroids
cent_data = torch.load('checkpoints/potato_ood_centroids.pt', map_location='cpu')
potato_centroid = cent_data['potato_centroid'].unsqueeze(0)
non_potato_centroid = cent_data['non_potato_centroid'].unsqueeze(0)
print(f"Loaded centroids from checkpoints/potato_ood_centroids.pt successfully!", flush=True)

# Define OOD classification decision rule
def is_potato_leaf(img):
    # Rule 1: Visual variance check (blank/flat images)
    arr = np.array(img.convert('RGB'))
    if float(np.std(arr)) < 8.0:
        return False, "low_variance", 0.0, 0.0

    # Rule 2: Foliage presence (leaves must have organic vegetation hues)
    r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    exg = 2*g - r - b
    # Check green or chlorotic/brown foliar content
    foliage_ratio = np.mean((exg > 5) | ((r > 60) & (g > 50) & (b < 100) & (r > b)))
    if foliage_ratio < 0.08:
        return False, "non_foliage", 0.0, float(foliage_ratio)

    # Rule 3: Deep Feature Domain Centroid Gate
    fn = F.normalize(extract_f(img), p=2, dim=1)
    sim_p = torch.matmul(fn, potato_centroid.t()).item()
    sim_np = torch.matmul(fn, non_potato_centroid.t()).item()
    margin = sim_p - sim_np

    # Decision boundary:
    # Real potato leaves have sim_p >= 0.48 and margin > 0.02
    is_valid = (sim_p >= 0.48) and (margin > 0.02)
    return is_valid, "domain_gate", sim_p, margin

print("\n--- 1. Evaluating ALL 28 Potato Samples ---", flush=True)
from app import CURATED_CLASS_SAMPLES
passed_p = 0
for c, files in CURATED_CLASS_SAMPLES.items():
    for f in files:
        p = os.path.join(DATA_DIR, c, f)
        if os.path.exists(p):
            ok, reason, s_p, m = is_potato_leaf(Image.open(p))
            if ok: passed_p += 1
            print(f"  Potato {c:12s} {f:18s} -> Valid: {str(ok):5s} | SimP: {s_p:6.4f} | Margin: {m:+6.4f}", flush=True)
print(f"Potato Pass Rate: {passed_p}/28 ({passed_p/28*100:.1f}%)", flush=True)

print("\n--- 2. Evaluating Non-Potato PlantVillage (Pepper & Tomato) ---", flush=True)
rejected_np = 0
total_np = 0
for folder in ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot", "Tomato_healthy", "Tomato_Early_blight"]:
    dir_p = os.path.join(PV_DIR, folder)
    if os.path.isdir(dir_p):
        for f in os.listdir(dir_p)[:3]:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_np += 1
                ok, reason, s_p, m = is_potato_leaf(Image.open(os.path.join(dir_p, f)))
                if not ok: rejected_np += 1
                print(f"  PV {folder[:20]:20s} {f[:12]} -> Valid: {str(ok):5s} | Reason: {reason:12s} | SimP: {s_p:6.4f} | Margin: {m:+6.4f}", flush=True)
print(f"Non-Potato Leaf Rejection Rate: {rejected_np}/{total_np} ({rejected_np/total_np*100:.1f}%)", flush=True)

print("\n--- 3. Evaluating Synthetic & Non-Leaf Inputs ---", flush=True)
synth = [
    ("Random Uniform Noise", Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))),
    ("Pure White Canvas", Image.fromarray(np.full((224, 224, 3), 255, dtype=np.uint8))),
    ("Pure Black Canvas", Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))),
    ("Pure Red Canvas", Image.fromarray(np.full((224, 224, 3), [220, 20, 20], dtype=np.uint8))),
    ("Pure Sky Blue Canvas", Image.fromarray(np.full((224, 224, 3), [30, 120, 230], dtype=np.uint8))),
    ("Color Gradient", Image.fromarray(np.linspace(0, 255, 224*224*3).reshape(224,224,3).astype(np.uint8))),
]
passed_synth = 0
for name, img in synth:
    ok, reason, s_p, m = is_potato_leaf(img)
    if not ok: passed_synth += 1
    print(f"  {name:22s} -> Valid: {str(ok):5s} | Reason: {reason:12s} | SimP: {s_p:6.4f} | Margin: {m:+6.4f}", flush=True)
print(f"Synthetic Rejection Rate: {passed_synth}/{len(synth)} ({passed_synth/len(synth)*100:.1f}%)", flush=True)

