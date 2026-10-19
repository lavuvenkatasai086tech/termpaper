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

# Let's find images in the system (e.g. IDE artifacts, screenshots, etc.)
artifact_dir = r"C:\Users\lavuv\.gemini\antigravity-ide\brain\3eb56759-7ef0-444b-9936-e2613c88e2bd"
imgs = []
for root, dirs, files in os.walk(artifact_dir):
    for f in files:
        if f.endswith('.png') or f.endswith('.jpg'):
            imgs.append((f, os.path.join(root, f)))
    if len(imgs) >= 5:
        break

print(f"Testing {len(imgs)} real screenshots/UI images (Non-potato objects):", flush=True)
for name, p in imgs[:5]:
    try:
        img = Image.open(p).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            eff_feats = model.effnet(t)
            vit_out = model.vit(t)
            vit_cls = vit_out.last_hidden_state[:, 0, :]
            vit_proj = model.vit_linear(vit_cls)
            f = torch.cat([eff_feats, vit_proj], dim=1)
            logits = model.classifier(f)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            energy = - torch.logsumexp(logits, dim=-1).item()
        top_c = int(np.argmax(probs))
        top_p = float(probs[top_c])
        print(f"  {name[:28]:28s} | Top: {top_c} ({top_p*100:5.1f}%) | Energy: {energy:6.3f} | Logits: {[round(x,2) for x in logits.squeeze().tolist()]}", flush=True)
    except Exception as e:
        print(f"Error {name}: {e}")
