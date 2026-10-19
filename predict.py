import os
import argparse
import torch
from PIL import Image
from dataset import get_transforms
from model import EfficientNetV2B3ViT


DEFAULT_ENERGY_THRESHOLD = 0.50
DEFAULT_TEMPERATURE = 1.0


def compute_energy_score(logits, temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    Computes non-Softmax Energy-Based Out-of-Distribution (OOD) score:
    Energy(x) = - T * logsumexp(logits / T)
    Lower free energy indicates high density / in-distribution potato sample.
    Higher free energy (exceeding threshold) indicates out-of-distribution or corrupted input.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    scaled_logits = logits / temperature
    energy = - temperature * torch.logsumexp(scaled_logits, dim=-1)
    return float(energy.item() if energy.dim() == 0 else energy[0].item())


OOD_CENTROIDS_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "potato_ood_centroids.pt")
ood_centroids_data = None

STAGE1_FILTER_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "potato_binary_filter.pth")
stage1_model_data = None

class CosineClassifier(torch.nn.Module):
    def __init__(self, in_features=576, num_classes=2, scale=12.0):
        super().__init__()
        self.scale = scale
        self.fc1 = torch.nn.Linear(in_features, 512)
        self.act = torch.nn.Hardswish()
        self.drop = torch.nn.Dropout(0.2)
        self.weight = torch.nn.Parameter(torch.randn(num_classes, 512))

    def forward(self, x):
        import torch.nn.functional as F
        h = self.drop(self.act(self.fc1(x)))
        h_norm = F.normalize(h, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * torch.matmul(h_norm, w_norm.t())

class Stage1Filter(torch.nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        f = torch.flatten(self.backbone(x), 1)
        return self.classifier(f)

def load_stage1_filter():
    global stage1_model_data
    if stage1_model_data is None and os.path.exists(STAGE1_FILTER_PATH):
        try:
            from torchvision import models
            ckpt = torch.load(STAGE1_FILTER_PATH, map_location="cpu")
            base = models.mobilenet_v3_small(weights=None)
            backbone = torch.nn.Sequential(base.features, base.avgpool)
            if "backbone_state_dict" in ckpt:
                backbone.load_state_dict(ckpt["backbone_state_dict"])
            backbone.eval()

            classifier = CosineClassifier()
            if "classifier_state_dict" in ckpt:
                classifier.load_state_dict(ckpt["classifier_state_dict"])
            classifier.eval()

            stage1_model_data = Stage1Filter(backbone, classifier)
            stage1_model_data.eval()
        except Exception as e:
            print(f"[Warning] Could not load Stage 1 filter: {e}")
    return stage1_model_data

def load_ood_centroids():
    global ood_centroids_data
    if ood_centroids_data is None and os.path.exists(OOD_CENTROIDS_PATH):
        try:
            ood_centroids_data = torch.load(OOD_CENTROIDS_PATH, map_location="cpu")
        except Exception:
            pass
    return ood_centroids_data


def predict_single_image(
    model,
    image_path: str,
    class_names: list,
    device: torch.device,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    temperature: float = DEFAULT_TEMPERATURE,
):
    """
    Predicts the disease class and probabilities for a given potato leaf image.
    Uses multi-factor Out-of-Distribution (OOD) detection combining botanical foliage verification,
    deep domain centroid similarity, and non-Softmax Energy scoring.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path).convert("RGB")
    import numpy as np
    import torch.nn.functional as F
    img_arr = np.array(image)

    # 1. Low pixel variance check
    if float(np.std(img_arr)) < 8.0:
        print("\n" + "=" * 65)
        print("[ALERT] Incorrect Input: Please upload a valid, clear image of a potato leaf.")
        print("  Reason: Low pixel variance canvas / blank image rejected.")
        print("=" * 65 + "\n")
        return {
            "is_valid": False,
            "is_ood": True,
            "error_message": "Invalid Image: Please upload a valid potato leaf image.",
            "predicted_class": None
        }

    # 2. Foliage presence check
    r = img_arr[:, :, 0].astype(float)
    g = img_arr[:, :, 1].astype(float)
    b = img_arr[:, :, 2].astype(float)
    exg = 2 * g - r - b
    foliage_mask = (exg > 5) | ((r > 60) & (g > 50) & (b < 100) & (r > b))
    foliage_ratio = float(np.mean(foliage_mask))
    if foliage_ratio < 0.08:
        print("\n" + "=" * 65)
        print("[ALERT] Incorrect Input: Please upload a valid, clear image of a potato leaf.")
        print(f"  Reason: Insufficient organic foliar content ({foliage_ratio:.3f} < 0.08). Non-plant input rejected.")
        print("=" * 65 + "\n")
        return {
            "is_valid": False,
            "is_ood": True,
            "error_message": "Invalid Image: Please upload a valid potato leaf image.",
            "predicted_class": None
        }

    # 3. STAGE 1: Lightweight Binary Plant Filter (Near-OOD Detection)
    stage1 = load_stage1_filter()
    if stage1 is not None:
        from torchvision import transforms
        s1_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        s1_tensor = s1_transform(image).unsqueeze(0)
        with torch.no_grad():
            s1_logits = stage1(s1_tensor)
            s1_probs = F.softmax(s1_logits, dim=1).squeeze().numpy()
        p_prob = float(s1_probs[0])
        np_prob = float(s1_probs[1])

        if p_prob < 0.65 or np_prob >= 0.35:
            print("\n" + "=" * 70)
            print("[ALERT] Incorrect Input: The uploaded image appears to be a different plant or non-potato crop. Please upload a valid potato leaf image.")
            print(f"  Stage 1 Decision: Rejected (Non-Potato Prob: {np_prob*100:.1f}%, Potato Prob: {p_prob*100:.1f}%)")
            print("  The 7-class potato disease prediction output has been suppressed.")
            print("=" * 70 + "\n")
            return {
                "is_valid": False,
                "is_ood": True,
                "ood_stage": "Stage 1 (Near-OOD Plant Filter)",
                "error_message": "Invalid Image: Please upload a valid potato leaf image.",
                "potato_probability": round(p_prob * 100, 2),
                "non_potato_probability": round(np_prob * 100, 2),
                "predicted_class": None,
                "probabilities": None
            }

    # STAGE 2: Hybrid 7-Class Disease Model
    transform = get_transforms(augment=False)
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        eff_f = model.effnet(tensor)
        v_out = model.vit(tensor)
        v_cls = v_out.last_hidden_state[:, 0, :]
        v_proj = model.vit_linear(v_cls)
        f = torch.cat([eff_f, v_proj], dim=1) # [1, 2048]

        logits = model.classifier(f)
        energy_score = compute_energy_score(logits, temperature=temperature)
        probabilities = torch.softmax(logits, dim=-1).cpu().squeeze().numpy()

        sim_p = 0.60
        margin = 0.20
        centroids = load_ood_centroids()
        if centroids is not None:
            f_norm = F.normalize(f, p=2, dim=1).cpu()
            p_cent = centroids["potato_centroid"].unsqueeze(0)
            np_cent = centroids["non_potato_centroid"].unsqueeze(0)
            sim_p = torch.matmul(f_norm, p_cent.t()).item()
            sim_np = torch.matmul(f_norm, np_cent.t()).item()
            margin = sim_p - sim_np

    top_idx = int(probabilities.argmax())
    predicted_class = class_names[top_idx]
    confidence = float(probabilities[top_idx])
    max_prob_pct = confidence * 100

    print(f"\nImage: {image_path}")
    print(f"Potato Sim: {sim_p:.4f} | Margin: {margin:+.4f} | Energy: {energy_score:.4f} (Threshold: {energy_threshold:.4f})")
    print(f"Top Softmax Probability: {max_prob_pct:.2f}% (Class: {predicted_class})")

    # Multi-factor Out-of-Distribution (OOD) Check
    is_ood = bool((sim_p < 0.48) or (margin < 0.02) or (energy_score > energy_threshold))

    if is_ood:
        print("\n" + "=" * 65)
        print("[ALERT] Incorrect Input: Please upload a valid, clear image of a potato leaf.")
        print(f"  Reason: Non-potato specimen rejected (sim_p={sim_p:.4f}, margin={margin:+.4f}, energy={energy_score:.4f}).")
        print("  The 7-class potato disease prediction output has been suppressed.")
        print("=" * 65 + "\n")
        return {
            "is_valid": False,
            "is_ood": True,
            "error_message": "Invalid Image: Please upload a valid potato leaf image.",
            "energy_score": energy_score,
            "energy_threshold": energy_threshold,
            "potato_similarity": sim_p,
            "foliar_margin": margin,
            "confidence": confidence,
            "predicted_class": None,
            "probabilities": None
        }

    print(f"Prediction: {predicted_class} ({max_prob_pct:.2f}%) [Valid Potato Specimen]")

    print("-" * 40)
    print("Class Probabilities:")
    for name, prob in sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {prob * 100:6.2f}%")

    return {
        "is_valid": True,
        "is_ood": False,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "energy_score": energy_score,
        "energy_threshold": energy_threshold,
        "probabilities": probabilities
    }


def main():
    parser = argparse.ArgumentParser(description="Inference on Potato Leaf Images with Energy-Based OOD Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--threshold", "--energy_threshold", dest="energy_threshold", type=float, default=DEFAULT_ENERGY_THRESHOLD, help="Energy score threshold for OOD rejection (default: 0.50)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature scaling factor for Energy score (default: 1.0)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint.get("class_names", [
        "Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"
    ])

    model = EfficientNetV2B3ViT(num_classes=len(class_names)).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    predict_single_image(
        model,
        args.image,
        class_names,
        device,
        energy_threshold=args.energy_threshold,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
