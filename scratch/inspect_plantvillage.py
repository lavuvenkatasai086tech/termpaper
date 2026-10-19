import os
import sys
sys.path.insert(0, r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project")
import torch
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

def inspect_image(path, name):
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        eff_feats = model.effnet(t)
        vit_out = model.vit(t)
        vit_cls = vit_out.last_hidden_state[:, 0, :]
        vit_proj = model.vit_linear(vit_cls)
        f = torch.cat([eff_feats, vit_proj], dim=1) # [1, 2048]
        logits = model.classifier(f)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        energy = - torch.logsumexp(logits, dim=-1).item()
        
    top_c = int(np.argmax(probs))
    top_p = float(probs[top_c])
    print(f"{name:35s} | Top: {top_c} ({top_p*100:5.1f}%) | Energy: {energy:6.3f} | Logits: {[round(x, 2) for x in logits.squeeze().tolist()]}")

print("=== PLANTVILLAGE NON-POTATO LEAVES (PEPPER & TOMATO) ===")
pv_root = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\PlantVillage"
for folder in ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot", "Tomato_healthy", "Tomato_Early_blight"]:
    f_path = os.path.join(pv_root, folder)
    if os.path.isdir(f_path):
        imgs = [os.path.join(f_path, x) for x in os.listdir(f_path) if x.endswith('.JPG') or x.endswith('.jpg')][:3]
        for idx, img_p in enumerate(imgs):
            inspect_image(img_p, f"{folder} #{idx+1}")
