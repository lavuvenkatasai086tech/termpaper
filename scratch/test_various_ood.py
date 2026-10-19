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

def test_img(img, name):
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
    print(f"{name:32s} | TopClass: {top_c} | Conf: {top_p*100:5.1f}% | Energy: {energy:6.3f} | Logits: {[round(x, 2) for x in logits.squeeze().tolist()]}")

print("=== TESTING NON-POTATO IMAGES (PLANTVILLAGE & OBJECTS) ===")
pv_root = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\PlantVillage"
for d in os.listdir(pv_root):
    if "Potato" in d: continue
    full_d = os.path.join(pv_root, d)
    if os.path.isdir(full_d):
        files = [f for f in os.listdir(full_d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            img = Image.open(os.path.join(full_d, files[0])).convert('RGB')
            test_img(img, d[:30])
