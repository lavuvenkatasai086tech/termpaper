import os
import requests

SERVER_URL = "http://127.0.0.1:5000"

print("--- 1. Testing Live API with NON-POTATO PlantVillage Leaves ---")
PV_DIR = r"c:\Users\lavuv\OneDrive\Desktop\Termpaper Project\PlantVillage"
for folder in ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot", "Tomato_Early_blight"]:
    dir_p = os.path.join(PV_DIR, folder)
    if os.path.isdir(dir_p):
        files = [x for x in os.listdir(dir_p) if x.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not files: continue
        full_p = os.path.join(dir_p, files[0])
        with open(full_p, 'rb') as fp:
            res = requests.post(f"{SERVER_URL}/api/predict", files={"file": fp})
        data = res.json()
        print(f"  {folder:25s} -> Status: {data.get('status')} | is_valid: {data.get('is_valid')} | is_ood: {data.get('is_ood')} | Error: {data.get('error_message')}")


print("\n--- 2. Testing Live API with NON-LEAF Synthetic Image ---")
import numpy as np
from PIL import Image
import io

noise_arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(noise_arr)
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)
res = requests.post(f"{SERVER_URL}/api/predict", files={"file": ("noise.jpg", buf, "image/jpeg")})
data = res.json()
print(f"  Random Uniform Noise      -> Status: {data.get('status')} | is_valid: {data.get('is_valid')} | is_ood: {data.get('is_ood')} | Error: {data.get('error_message')}")

print("\n--- 3. Testing Live API with Curated Potato Samples ---")
for sample_path in [
    "Bacteria/1692332350433.jpg",
    "Fungi/20230712_150159.jpg",
    "Healthy/1692336758850.jpg",
    "Nematode/20230712_120909.jpg",
    "Pest/1692332350400.jpg",
    "Phytopthora/20230712_155047.jpg",
    "Virus/1692335404669.jpg"
]:
    res = requests.post(f"{SERVER_URL}/api/predict", data={"sample_path": sample_path})
    data = res.json()
    print(f"  Potato {sample_path:30s} -> Class: {data.get('predicted_class')} | Conf: {data.get('confidence')}% | is_valid: {data.get('is_valid')} | is_ood: {data.get('is_ood')}")
