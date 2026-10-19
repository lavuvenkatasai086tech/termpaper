import requests, glob, os

url = 'http://127.0.0.1:5000/api/predict'

# 1. Test Near-OOD Crops
crops = [
    ('Tomato', 'PlantVillage/Tomato_Leaf_Mold'),
    ('Pepper', 'PlantVillage/Pepper__bell___healthy'),
    ('Apple', 'PlantVillage/Apple___healthy'),
    ('Corn', 'PlantVillage/Corn___healthy'),
    ('Grape', 'PlantVillage/Grape___healthy'),
]

print("=== TESTING NEAR-OOD CROPS (Stage 1 Binary Filter) ===")
for name, folder in crops:
    files = glob.glob(f"{folder}/*.*")
    if files:
        fpath = files[0]
        with open(fpath, 'rb') as fp:
            res = requests.post(url, files={'file': fp})
        data = res.json()
        is_ood = data.get('is_ood')
        err = data.get('error_message')
        print(f"[{name:8s}] is_ood: {is_ood} | Alert: {err}")

print("\n=== TESTING VALID POTATO SAMPLES (Stage 2 Disease Prediction) ===")
test_potatoes = [
    ('Bacteria', 'data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Bacteria/1692332350433.jpg'),
    ('Healthy', 'data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Healthy/1692336758850.jpg'),
    ('Late Blight', 'PlantVillage/Potato___Late_blight/' + os.listdir('PlantVillage/Potato___Late_blight')[0])
]

for label, fpath in test_potatoes:
    with open(fpath, 'rb') as fp:
        res = requests.post(url, files={'file': fp})
    data = res.json()
    is_ood = data.get('is_ood')
    pred = data.get('predicted_class')
    conf = data.get('confidence')
    print(f"[{label:12s}] is_ood: {is_ood} | Prediction: {pred} ({conf}%)")
