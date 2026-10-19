import urllib.request
import urllib.parse
import json
import os

pv_crops = [
    ("Tomato_Leaf_Mold", "Tomato Leaf (Near-OOD)"),
    ("Corn_maize___Northern_Leaf_Blight", "Corn Leaf (Near-OOD)"),
    ("Apple___healthy", "Apple Leaf (Near-OOD)"),
    ("Orange___Haunglongbing_Citrus_greening", "Citrus Leaf (Near-OOD)"),
    ("Peach___healthy", "Peach Leaf (Near-OOD)"),
    ("Squash___Powdery_mildew", "Squash Leaf (Near-OOD)"),
    ("Outdoor_Non_Potato", "Outdoor Foliage (Non-Potato)"),
    ("Non_Plant_Objects", "Coffee Mug / Desk (Object)"),
]

pv_dir = r"PlantVillage"
for folder, label in pv_crops:
    folder_path = os.path.join(pv_dir, folder)
    if os.path.isdir(folder_path):
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for fname in files[:2 if folder in ("Outdoor_Non_Potato", "Non_Plant_Objects") else 1]:
            rel_path = f"PlantVillage/{folder}/{fname}"
            # test image fetch
            img_url = f"http://127.0.0.1:5000/api/sample-image/{urllib.parse.quote(rel_path)}"
            with urllib.request.urlopen(img_url) as resp:
                size = len(resp.read())
            # test predict
            payload = json.dumps({'sample_path': rel_path}).encode('utf-8')
            req = urllib.request.Request('http://127.0.0.1:5000/api/predict', data=payload, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            print(f"{label:28s} | {fname[:24]:24s} | {size:6d}B | OOD: {data.get('is_ood')} | Valid: {data.get('is_valid')}")
