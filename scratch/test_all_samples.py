import requests

resp = requests.get('http://127.0.0.1:5000/api/samples')
data = resp.json()
samples = data.get('samples', [])

total_potato = 0
passed_potato = 0
total_invalid = 0
rejected_invalid = 0

for s in samples:
    cls = s['class_name']
    rel_path = s['relative_path']
    r = requests.post('http://127.0.0.1:5000/api/predict', json={'sample_path': rel_path})
    res = r.json()
    is_ood = res.get('is_ood', False)
    pred = res.get('predicted_class')
    
    if cls == 'Invalid_OOD' or 'PlantVillage' in rel_path:
        total_invalid += 1
        if is_ood:
            rejected_invalid += 1
        else:
            print(f"FAILED to reject OOD: {rel_path} -> classified as {pred}")
    else:
        total_potato += 1
        if not is_ood and pred == cls:
            passed_potato += 1
        else:
            print(f"FAILED potato sample: {cls} / {rel_path} -> is_ood={is_ood}, pred={pred}")

print(f"\nPotato samples: {passed_potato}/{total_potato} ({passed_potato/total_potato*100:.1f}%)")
if total_invalid > 0:
    print(f"Invalid non-potato samples: {rejected_invalid}/{total_invalid} ({rejected_invalid/total_invalid*100:.1f}%) rejected!")
