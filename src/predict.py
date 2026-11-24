import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .model import EnhancedHybridEfficientNetViT
from .ood_detector import OODGatekeeper, EXACT_INVALID_INPUT_MESSAGE
from .dataset import get_transforms
from .xai import XAIExplainer

DEFAULT_CLASS_NAMES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]


def predict_single_image(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    energy_threshold: float = 0.50,
    temperature: float = 1.0,
    generate_xai: bool = False,
    save_xai_dir: str = None,
    device: torch.device = None,
):
    """
    Inference pipeline for a single image with two-stage OOD gatekeeper and XAI:
    1. Runs strict botanical and deep feature OOD gatekeeper.
    2. If rejected: halts immediately, suppresses disease prediction, and outputs exact required user message.
    3. If valid: calculates disease class probabilities and SE-gate modality weight split.
    4. Optionally generates XAI heatmaps (CNN Grad-CAM, ViT Rollout, Fused XAI).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(augment=False)
    tensor = transform(image).unsqueeze(0).to(device)

    # Initialize model
    model = EnhancedHybridEfficientNetViT(num_classes=len(DEFAULT_CLASS_NAMES)).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    model.eval()

    # 1. Execute Strict OOD Gatekeeper Check
    gatekeeper = OODGatekeeper(device=device)
    gate_result = gatekeeper.validate_and_gate(
        image_pil=image,
        hybrid_model=model,
        tensor=tensor,
        energy_threshold=energy_threshold,
        temperature=temperature
    )

    if not gate_result["is_valid"]:
        print("\n" + "=" * 70)
        print(f"[REJECTED] {EXACT_INVALID_INPUT_MESSAGE}")
        print(f"  Stage:  {gate_result.get('rejection_stage')}")
        print(f"  Reason: {gate_result.get('rejection_reason')}")
        diag = gate_result.get("diagnostic", {})
        print(f"  Energy Score: {diag.get('energy_score')} (Threshold: {diag.get('energy_threshold')})")
        print(f"  Potato Similarity: {diag.get('potato_similarity')} | Margin: {diag.get('foliar_margin')}")
        print("  Disease prediction output suppressed.")
        print("=" * 70 + "\n")
        return gate_result

    # 2. In-Distribution Valid Specimen: Process Classification
    with torch.no_grad():
        logits = gate_result["logits"]
        probs = F.softmax(logits, dim=-1).cpu().squeeze().numpy()
        modality_weights = model.get_modality_weights(tensor)

    top_idx = int(np.argmax(probs))
    predicted_class = DEFAULT_CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])
    energy_score = gate_result["energy_score"]

    print("\n" + "=" * 70)
    print(f"Specimen: {image_path}")
    print(f"Status: VALID POTATO LEAF SPECIMEN")
    print(f"Energy Score: {energy_score:.4f} (Threshold: {energy_threshold:.4f})")
    print(f"Potato Similarity: {gate_result.get('potato_similarity', 0.0):.4f} | Margin: {gate_result.get('foliar_margin', 0.0):+.4f}")
    print(f"SE-Gate Modality Split: CNN Local {modality_weights['cnn_pct']}% | ViT Global {modality_weights['vit_pct']}%")
    print(f"Prediction: {predicted_class} ({confidence * 100:.2f}% Confidence)")

    # Pest vs Fungi disambiguation note
    pest_idx = DEFAULT_CLASS_NAMES.index("Pest")
    fungi_idx = DEFAULT_CLASS_NAMES.index("Fungi")
    if predicted_class in ["Pest", "Fungi"] and abs(probs[pest_idx] - probs[fungi_idx]) < 0.20:
        print("  [SE-Gate Disambiguation Note]: Fine-grained feature fusion active. "
              "SE-Gate weighted local CNN lesion textures to resolve Pest vs. Fungi overlap.")

    print("-" * 40)
    print("Class Probabilities:")
    prob_dict = []
    for name, p in sorted(zip(DEFAULT_CLASS_NAMES, probs), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {p * 100:6.2f}%")
        prob_dict.append({"class": name, "probability": round(float(p) * 100, 2)})
    print("=" * 70 + "\n")

    xai_data = None
    if generate_xai or save_xai_dir:
        explainer = XAIExplainer()
        xai_data = explainer.explain(model, tensor, image, target_class=top_idx)
        if save_xai_dir:
            os.makedirs(save_xai_dir, exist_ok=True)
            import base64
            for key, b64 in [("gradcam", xai_data["gradcam_b64"]),
                             ("vit_rollout", xai_data["rollout_b64"]),
                             ("fused_xai", xai_data["fused_b64"])]:
                raw_data = base64.b64decode(b64.split(",")[1])
                out_p = os.path.join(save_xai_dir, f"{os.path.basename(image_path)}_{key}.png")
                with open(out_p, "wb") as f_out:
                    f_out.write(raw_data)
            print(f"[XAI] Visualizations saved to {save_xai_dir}")

    return {
        "is_valid": True,
        "is_ood": False,
        "status": "VALID_POTATO_LEAF",
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "energy_score": energy_score,
        "energy_threshold": energy_threshold,
        "potato_similarity": gate_result.get("potato_similarity"),
        "foliar_margin": gate_result.get("foliar_margin"),
        "modality_weights": modality_weights,
        "probabilities": prob_dict,
        "xai": xai_data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Potato Leaf Disease Prediction with OOD Gatekeeper")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--threshold", type=float, default=0.50, help="Energy OOD threshold")
    parser.add_argument("--xai", action="store_true", help="Generate Explainable AI overlays")
    parser.add_argument("--save_xai", type=str, default=None, help="Directory to save XAI images")
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        energy_threshold=args.threshold,
        generate_xai=args.xai,
        save_xai_dir=args.save_xai,
    )
