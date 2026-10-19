import os
from PIL import Image
import numpy as np

DATA_DIR = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\data\potatodata\Potato Leaf Disease Dataset in Uncontrolled Environment"
pv_root = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\PlantVillage"

def inspect_meta(path):
    img = Image.open(path)
    arr = np.array(img.convert('RGB'))
    h, w, c = arr.shape
    mean_rgb = arr.mean(axis=(0,1))
    std_rgb = arr.std(axis=(0,1))
    # ExG (Excess Green index) = 2*G - R - B
    r, g, b = arr[:,:,0].astype(float), arr[:,:,1].astype(float), arr[:,:,2].astype(float)
    exg = 2*g - r - b
    green_pct = np.mean(exg > 15) * 100
    return h, w, mean_rgb, std_rgb, green_pct

print("--- POTATO DATASET SAMPLES ---")
for c in ["Bacteria", "Healthy", "Phytopthora", "Virus"]:
    p = os.path.join(DATA_DIR, c)
    files = [f for f in os.listdir(p) if f.lower().endswith('.jpg')][:2]
    for f in files:
        h, w, m, s, gp = inspect_meta(os.path.join(p, f))
        print(f"Potato {c:12s} ({f:18s}): Size {w}x{h} | MeanRGB: {[round(x,1) for x in m]} | ExG>15%: {gp:4.1f}%")

print("\n--- PLANTVILLAGE NON-POTATO SAMPLES ---")
for c in ["Pepper__bell___healthy", "Tomato_healthy", "Tomato_Early_blight"]:
    p = os.path.join(pv_root, c)
    files = [f for f in os.listdir(p) if f.lower().endswith('.jpg') or f.lower().endswith('.JPG')][:2]
    for f in files:
        h, w, m, s, gp = inspect_meta(os.path.join(p, f))
        print(f"PV {c[:20]:20s} ({f[:15]}): Size {w}x{h} | MeanRGB: {[round(x,1) for x in m]} | ExG>15%: {gp:4.1f}%")
