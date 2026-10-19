import os
import glob
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

torch.manual_seed(42)
random.seed(42)

POTATO_DIRS = [
    "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Bacteria",
    "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Fungi",
    "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Healthy",
    "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Nematode",
    "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Pest",
    "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Phytopthora",
    "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Virus",
    "PlantVillage/Potato___Late_blight",
    "PlantVillage/Potato___healthy",
]

NON_POTATO_DIRS = [
    "PlantVillage/Cotton_leaves",
    "PlantVillage/Outdoor_Non_Potato",
    "PlantVillage/Apple___Apple_scab",
    "PlantVillage/Apple___Black_rot",
    "PlantVillage/Apple___healthy",
    "PlantVillage/Corn___healthy",
    "PlantVillage/Corn_maize___Common_rust_",
    "PlantVillage/Corn_maize___Northern_Leaf_Blight",
    "PlantVillage/Grape___Black_rot",
    "PlantVillage/Grape___Leaf_blight_Isariopsis_Leaf_Spot",
    "PlantVillage/Grape___healthy",
    "PlantVillage/Peach___Bacterial_spot",
    "PlantVillage/Peach___healthy",
    "PlantVillage/Cherry_including_sour___healthy",
    "PlantVillage/Orange___Haunglongbing_Citrus_greening",
    "PlantVillage/Soybean___healthy",
    "PlantVillage/Squash___Powdery_mildew",
    "PlantVillage/Strawberry___healthy",
    "PlantVillage/Pepper__bell___healthy",
    "PlantVillage/Tomato_Leaf_Mold",
    "PlantVillage/Tomato__Target_Spot",
    "PlantVillage/Tomato__Tomato_YellowLeaf__Curl_Virus",
]

def load_dataset():
    potato_files = []
    for d in POTATO_DIRS:
        if os.path.exists(d):
            imgs = [f for f in glob.glob(os.path.join(d, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(imgs)
            potato_files.extend(imgs[:45]) # ~400 potato images

    non_potato_files = []
    for d in NON_POTATO_DIRS:
        if os.path.exists(d):
            imgs = [f for f in glob.glob(os.path.join(d, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # If cotton or outdoor non-potato, repeat/augment them so they have strong representation
            if "Cotton" in d or "Outdoor" in d:
                imgs = imgs * 4
            random.shuffle(imgs)
            non_potato_files.extend(imgs[:25]) # ~400 non-potato images

    print(f"Loaded {len(potato_files)} potato images and {len(non_potato_files)} non-potato images across {len(NON_POTATO_DIRS)} non-potato classes.")

    data = [(p, 0) for p in potato_files] + [(p, 1) for p in non_potato_files]
    random.shuffle(data)

    split = int(0.85 * len(data))
    return data[:split], data[split:]

class CosineClassifier(nn.Module):
    def __init__(self, in_features=576, num_classes=2, scale=12.0):
        super().__init__()
        self.scale = scale
        self.fc1 = nn.Linear(in_features, 512)
        self.act = nn.Hardswish()
        self.drop = nn.Dropout(0.2)
        self.weight = nn.Parameter(torch.randn(num_classes, 512))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        h = self.drop(self.act(self.fc1(x)))
        h_norm = F.normalize(h, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        logits = self.scale * torch.matmul(h_norm, w_norm.t())
        return logits

def extract_features(model_backbone, data_list, transform, device):
    feats = []
    labels = []
    t0 = time.time()
    with torch.no_grad():
        for i, (path, label) in enumerate(data_list):
            try:
                img = Image.open(path).convert("RGB")
                t = transform(img).unsqueeze(0).to(device)
                f = model_backbone(t)
                f = torch.flatten(f, 1)
                feats.append(f.cpu())
                labels.append(label)
            except Exception as e:
                pass
    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)

def main():
    device = torch.device("cpu")
    print(f"Training Robust MobileNetV3 Stage 1 Filter on: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data, val_data = load_dataset()

    base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    base_model.eval()
    backbone = nn.Sequential(base_model.features, base_model.avgpool).to(device)

    train_feats, train_labels = extract_features(backbone, train_data, transform, device)
    val_feats, val_labels = extract_features(backbone, val_data, transform, device)

    classifier = CosineClassifier(in_features=576, num_classes=2, scale=12.0).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.003, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    num_samples = len(train_labels)
    batch_size = 32

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, 35):
        classifier.train()
        perm = torch.randperm(num_samples)

        for i in range(0, num_samples, batch_size):
            idxs = perm[i:i+batch_size]
            b_feats, b_labels = train_feats[idxs], train_labels[idxs]

            optimizer.zero_grad()
            preds = classifier(b_feats)
            loss = criterion(preds, b_labels)
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_preds = classifier(val_feats)
            val_acc = (val_preds.argmax(dim=1) == val_labels).float().mean().item()
            train_acc = (classifier(train_feats).argmax(dim=1) == train_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = classifier.state_dict()

        if epoch % 5 == 0 or epoch == 34:
            print(f"Epoch {epoch:2d} | Train Acc: {train_acc*100:5.1f}% | Val Acc: {val_acc*100:5.1f}%")

    # Save the robust filter
    save_path = "checkpoints/potato_binary_filter.pth"
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "classifier_state_dict": best_weights,
        "classifier_type": "cosine",
        "in_features": 576,
        "num_classes": 2,
        "val_accuracy": best_val_acc,
        "classes": ["Potato Leaf", "Non-Potato / Other Plant Leaf"],
    }, save_path)
    print(f"\nSaved Best Robust Binary Filter to: {save_path} (Val Acc: {best_val_acc*100:.2f}%)")

    # TEST AGAINST THE 5 USER UPLOADED IMAGES
    print("\n" + "=" * 70)
    print("TESTING USER UPLOADED IMAGES WITH ROBUST BINARY FILTER:")
    print("=" * 70)

    classifier.load_state_dict(best_weights)
    classifier.eval()

    user_dir = "C:/Users/lavuv/.gemini/antigravity-ide/brain/3eb56759-7ef0-444b-9936-e2613c88e2bd/.user_uploaded"
    user_files = sorted(glob.glob(f"{user_dir}/*.*"))

    for f in user_files:
        img = Image.open(f).convert("RGB")
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = torch.flatten(backbone(t), 1)
            logits = classifier(feat)
            probs = F.softmax(logits, dim=1).squeeze().numpy()
        p_prob = probs[0]
        np_prob = probs[1]
        decision = "PASS (Potato)" if p_prob >= 0.70 else "REJECTED (Non-Potato / Invalid Input)"
        print(f"  {os.path.basename(f):25s} -> Potato: {p_prob*100:5.1f}%, Non-Potato: {np_prob*100:5.1f}% [{decision}]")

    print("\n" + "=" * 70)
    print("TESTING 28 CURATED POTATO SAMPLES:")
    print("=" * 70)
    import app
    passed = 0
    total = 0
    for cls, files in app.CURATED_CLASS_SAMPLES.items():
        for f in files:
            p = os.path.join(app.DATA_DIR, cls, f)
            if os.path.exists(p):
                img = Image.open(p).convert("RGB")
                t = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = torch.flatten(backbone(t), 1)
                    logits = classifier(feat)
                    probs = F.softmax(logits, dim=1).squeeze().numpy()
                p_prob = probs[0]
                status = "PASS" if p_prob >= 0.70 else "FAIL"
                if p_prob >= 0.70: passed += 1
                total += 1
                print(f"  [{cls:12s}] {f}: Potato={p_prob*100:5.1f}% [{status}]")

    print(f"\nCurated Potato Samples Result: {passed}/{total} ({passed/total*100:.1f}%) PASSED as Potato Leaf!")

if __name__ == "__main__":
    main()
