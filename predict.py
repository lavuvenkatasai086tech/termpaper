import os
import argparse
import torch
from PIL import Image
from dataset import get_transforms
from model import EfficientNetV2B3ViT


def predict_single_image(model, image_path, class_names, device):
    """
    Predicts the disease class and probabilities for a given leaf image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    transform = get_transforms(augment=False)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probabilities = model.predict_proba(tensor).cpu().squeeze().numpy()

    top_idx = probabilities.argmax()
    predicted_class = class_names[top_idx]
    confidence = probabilities[top_idx]

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class} ({confidence * 100:.2f}%)")
    print("-" * 40)
    print("Class Probabilities:")
    for name, prob in sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {prob * 100:6.2f}%")

    return predicted_class, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(description="Inference on Potato Leaf Images")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Model checkpoint path")
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

    predict_single_image(model, args.image, class_names, device)


if __name__ == "__main__":
    main()
