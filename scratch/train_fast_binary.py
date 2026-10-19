import os
import glob
import random
import time
import torch
import torch.nn as nn
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
    "PlantVillage/Tomato_Leaf_Mold",
    "PlantVillage/Tomato__Target_Spot",
    "PlantVillage/Tomato__Tomato_YellowLeaf__Curl_Virus",
    "PlantVillage/Pepper__bell___healthy",
    "PlantVillage/Apple___healthy",
    "PlantVillage/Corn___healthy",
    "PlantVillage/Grape___healthy",
]

def load_balanced_data(n_per_class=250):
    potato_files = []
    for d in POTATO_DIRS:
        if os.path.exists(d):
            imgs = [f for f in glob.glob(os.path.join(d, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            # Take evenly from each disease class
            take = max(1, n_per_class // len(POTATO_DIRS))
            random.shuffle(imgs)
            potato_files.extend(imgs[:take])

    non_potato_files = []
    for d in NON_POTATO_DIRS:
        if os.path.exists(d):
            imgs = [f for f in glob.glob(os.path.join(d, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            take = max(1, n_per_class // len(NON_POTATO_DIRS))
            random.shuffle(imgs)
            non_potato_files.extend(imgs[:take])

    print(f"Collected {len(potato_files)} potato images and {len(non_potato_files)} non-potato images.")

    # Shuffle
    random.shuffle(potato_files)
    random.shuffle(non_potato_files)

    data = [(p, 0) for p in potato_files] + [(p, 1) for p in non_potato_files]
    random.shuffle(data)

    split = int(0.80 * len(data))
    return data[:split], data[split:]

def extract_features(model_backbone, data_list, transform, device):
    feats = []
    labels = []
    t0 = time.time()
    print(f"Extracting features for {len(data_list)} images...")
    with torch.no_grad():
        for i, (path, label) in enumerate(data_list):
            try:
                img = Image.open(path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                f = model_backbone(tensor) # [1, 576, 1, 1]
                f = torch.flatten(f, 1) # [1, 576]
                feats.append(f.cpu())
                labels.append(label)
            except Exception as e:
                print(f"Error loading {path}: {e}")
            if (i + 1) % 100 == 0 or (i + 1) == len(data_list):
                print(f"  {i+1}/{len(data_list)} extracted in {time.time()-t0:.1f}s")
    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.long)

def main():
    device = torch.device("cpu")
    print(f"Training Stage 1 Binary Classifier on: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data, val_data = load_balanced_data(n_per_class=250)

    # Base MobileNetV3-Small
    base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    base_model.eval()

    # Backbone is features + avgpool
    backbone = nn.Sequential(base_model.features, base_model.avgpool).to(device)

    train_feats, train_labels = extract_features(backbone, train_data, transform, device)
    val_feats, val_labels = extract_features(backbone, val_data, transform, device)

    # Classifier head: matches MobileNetV3 classifier architecture
    classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 2)
    )

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.002, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\nTraining binary classifier head...")
    batch_size = 32
    num_samples = len(train_labels)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, 25):
        classifier.train()
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0

        for i in range(0, num_samples, batch_size):
            idxs = perm[i:i+batch_size]
            b_feats, b_labels = train_feats[idxs], train_labels[idxs]

            optimizer.zero_grad()
            preds = classifier(b_feats)
            loss = criterion(preds, b_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(b_labels)

        # Validation
        classifier.eval()
        with torch.no_grad():
            val_preds = classifier(val_feats)
            val_acc = (val_preds.argmax(dim=1) == val_labels).float().mean().item()
            train_acc = (classifier(train_feats).argmax(dim=1) == train_labels).float().mean().item()

        if epoch % 5 == 0 or epoch == 24:
            print(f"Epoch {epoch:2d} | Train Acc: {train_acc*100:5.1f}% | Val Acc: {val_acc*100:5.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = classifier.state_dict()

    # Construct the full MobileNetV3-Small with the trained classifier
    final_model = models.mobilenet_v3_small(weights=None)
    final_model.classifier = classifier
    final_model.load_state_dict({
        **{f"features.{k}": v for k, v in base_model.features.state_dict().items()},
        **{f"classifier.{k}": v for k, v in best_weights.items()}
    }, strict=False)

    save_path = "checkpoints/potato_binary_filter.pth"
    torch.save({
        "model_state_dict": final_model.state_dict(),
        "val_accuracy": best_val_acc,
        "classes": ["Potato Leaf", "Non-Potato / Other Plant Leaf"],
        "class_to_idx": {"Potato Leaf": 0, "Non-Potato / Other Plant Leaf": 1}
    }, save_path)
    print(f"\nSaved Best Binary Filter to: {save_path} (Val Acc: {best_val_acc*100:.2f}%)")

    # Evaluation on Diverse Crops
    print("\n" + "=" * 70)
    print("STAGE 1 BINARY FILTER EVALUATION ON NEAR-OOD CROPS & POTATO LEAVES")
    print("=" * 70)

    final_model.eval()
    test_crops = [
        ("Potato (Bacteria)", "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Bacteria"),
        ("Potato (Healthy)", "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Healthy"),
        ("Potato (Late Blight)", "PlantVillage/Potato___Late_blight"),
        ("Tomato (Leaf Mold)", "PlantVillage/Tomato_Leaf_Mold"),
        ("Tomato (Yellow Leaf Curl)", "PlantVillage/Tomato__Tomato_YellowLeaf__Curl_Virus"),
        ("Pepper (Healthy)", "PlantVillage/Pepper__bell___healthy"),
        ("Apple (Healthy)", "PlantVillage/Apple___healthy"),
        ("Corn (Healthy)", "PlantVillage/Corn___healthy"),
        ("Grape (Healthy)", "PlantVillage/Grape___healthy"),
    ]

    for crop_name, folder in test_crops:
        if os.path.exists(folder):
            files = [f for f in glob.glob(os.path.join(folder, "*.*")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
            if not files:
                continue
            potato_scores = []
            non_potato_scores = []
            for f in files:
                img = Image.open(f).convert("RGB")
                t = transform(img).unsqueeze(0)
                with torch.no_grad():
                    logits = final_model(t)
                    probs = torch.softmax(logits, dim=1).squeeze().numpy()
                potato_scores.append(probs[0])
                non_potato_scores.append(probs[1])

            avg_p = sum(potato_scores) / len(potato_scores)
            avg_np = sum(non_potato_scores) / len(non_potato_scores)
            decision = "ACCEPTED as Potato" if avg_p >= 0.50 else "REJECTED as Non-Potato"
            print(f"  {crop_name:28s} -> Potato Prob: {avg_p*100:5.1f}% | Non-Potato: {avg_np*100:5.1f}% [{decision}]")

if __name__ == "__main__":
    main()
