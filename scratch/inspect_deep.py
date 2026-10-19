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

# Let's inspect ViT attention / feature statistics or EffNet layer statistics
def analyze_img(path):
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        eff_f = model.effnet(t) # [1, 1536]
        vit_out = model.vit(t)
        vit_cls = vit_out.last_hidden_state[:, 0, :] # [1, 768]
        # Also check all patch tokens from ViT!
        patch_tokens = vit_out.last_hidden_state[:, 1:, :] # [1, 196, 768]
        patch_std = patch_tokens.std(dim=1).mean().item()
        patch_mean = patch_tokens.mean().item()
        
        logits = model.classifier(torch.cat([eff_f, model.vit_linear(vit_cls)], dim=1))
        energy = - torch.logsumexp(logits, dim=-1).item()
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        top_p = float(probs.max())
        
        # Color stats
        arr = np.array(img)
        r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
        exg = 2*g - r - b
        foliage_ratio = np.mean(exg > 10)
        
    return eff_f.norm().item(), vit_cls.norm().item(), patch_std, energy, top_p, foliage_ratio

print("POTATO DATASET SAMPLES:")
for c in ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]:
    folder = os.path.join(DATA_DIR, c)
    files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')][:3]
    for f in files:
        ef_n, vt_n, p_std, en, tp, fol = analyze_img(os.path.join(folder, f))
        print(f"Potato {c:12s} | EffNorm: {ef_n:5.2f} | ViTNorm: {vt_n:5.2f} | PatchStd: {p_std:5.3f} | En: {en:6.3f} | Conf: {tp*100:5.1f}% | Foliage: {fol*100:4.1f}%")

print("\nPLANTVILLAGE NON-POTATO (Pepper & Tomato):")
for c in ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot", "Tomato_healthy", "Tomato_Early_blight"]:
    folder = os.path.join(PV_DIR, c)
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.JPG'))][:3]
    for f in files:
        ef_n, vt_n, p_std, en, tp, fol = analyze_img(os.path.join(folder, f))
        print(f"PV {c[:20]:20s} | EffNorm: {ef_n:5.2f} | ViTNorm: {vt_n:5.2f} | PatchStd: {p_std:5.3f} | En: {en:6.3f} | Conf: {tp*100:5.1f}% | Foliage: {fol*100:4.1f}%")
