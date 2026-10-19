import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# Set random seeds for reproducibility
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

class BinaryLeafDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def collect_samples(max_per_group=1500):
    potato_files = []
    for d in POTATO_DIRS:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.*"))
            potato_files.extend([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    non_potato_files = []
    for d in NON_POTATO_DIRS:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.*"))
            non_potato_files.extend([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"Total found: {len(potato_files)} potato images, {len(non_potato_files)} non-potato images")

    random.shuffle(potato_files)
    random.shuffle(non_potato_files)

    selected_potato = potato_files[:max_per_group]
    selected_non_potato = non_potato_files[:max_per_group]

    samples = [(p, 0) for p in selected_potato] + [(p, 1) for p in selected_non_potato]
    random.shuffle(samples)

    split = int(0.85 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    print(f"Dataset prepared: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_samples, val_samples = collect_samples(max_per_group=1200)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(BinaryLeafDataset(train_samples, train_transform), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(BinaryLeafDataset(val_samples, val_transform), batch_size=32, shuffle=False, num_workers=0)

    # Load MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # Replace final linear layer for binary classification: 0 = Potato Leaf, 1 = Non-Potato Leaf
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model = model.to(device)

    # Freeze earlier layers, fine-tune later layers
    for param in model.features[:8].parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    epochs = 5
    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/potato_binary_filter.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * len(labels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch}/{epochs} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Loss: {val_loss/val_total:.4f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
                "classes": ["potato_leaf", "non_potato_leaf"],
                "class_to_idx": {"potato_leaf": 0, "non_potato_leaf": 1}
            }, save_path)
            print(f"  -> Saved best binary filter to {save_path} (Val Acc: {val_acc*100:.2f}%)")

    print("\nEvaluating against test specimens...")
    # Load best checkpoint
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Test Apple, Grape, Corn, Tomato, Pepper
    test_crops = [
        ("Apple", "PlantVillage/Apple___healthy"),
        ("Corn", "PlantVillage/Corn___healthy"),
        ("Grape", "PlantVillage/Grape___healthy"),
        ("Pepper", "PlantVillage/Pepper__bell___healthy"),
        ("Tomato", "PlantVillage/Tomato_Leaf_Mold"),
    ]

    print("\n--- Testing Non-Potato Leaves (Should predict Non-Potato, class 1) ---")
    for crop_name, folder in test_crops:
        if os.path.exists(folder):
            test_files = glob.glob(os.path.join(folder, "*.*"))[:5]
            for f in test_files:
                img = Image.open(f).convert("RGB")
                t = val_transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(t)
                    probs = torch.softmax(logits, dim=1).cpu().squeeze().numpy()
                    pred = int(probs.argmax())
                    is_potato = (pred == 0)
                status = "PASS (Rejected Non-Potato)" if not is_potato else "FAIL (Classified as Potato)"
                print(f"[{crop_name}] {os.path.basename(f)}: Non-Potato Prob = {probs[1]*100:.1f}%, Potato Prob = {probs[0]*100:.1f}% -> {status}")

if __name__ == "__main__":
    train()
