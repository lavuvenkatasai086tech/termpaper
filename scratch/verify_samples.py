import json, os, torch
from PIL import Image
from torchvision import transforms, models

ckpt = torch.load('checkpoints/potato_binary_filter.pth', map_location='cpu')
model = models.mobilenet_v3_small(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(576, 1024),
    torch.nn.Hardswish(),
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(1024, 2)
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

import sys
sys.path.insert(0, os.path.abspath("."))
import app
samples = app.SAMPLE_IMAGES
print(f"Testing {len(samples)} curated samples from app.py:")
passed = 0
failed = 0
for s in samples:
    path = os.path.join(app.DATA_DIR, s['path']) if not s['path'].startswith('PlantVillage') else s['path']
    if not os.path.exists(path):
        print(f"File missing: {path}")
        continue
    img = Image.open(path).convert('RGB')
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1).squeeze().numpy()
    p_prob = probs[0]
    is_potato = (p_prob >= 0.50)
    is_sample_ood = s.get('is_ood_sample', False)
    
    if is_sample_ood:
        status = 'CORRECTLY REJECTED OOD' if not is_potato else 'MISSED OOD'
    else:
        status = 'PASS (Potato)' if is_potato else 'FALSE REJECTION'
    
    if (is_sample_ood and not is_potato) or (not is_sample_ood and is_potato):
        passed += 1
    else:
        failed += 1
    cat = s['category']
    name = s['name']
    print(f"[{cat:15s}] {name:20s}: Potato={p_prob*100:5.1f}%, Non-Potato={probs[1]*100:5.1f}% -> {status}")

print(f"\nTotal Test Results: {passed}/{passed+failed} ({passed/(passed+failed)*100:.1f}%) matches desired outcome!")
