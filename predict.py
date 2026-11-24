"""
Root compatibility wrapper for predict.py.
Imports predict pipeline and OOD checks from src.predict.
"""
from src.predict import (
    predict_single_image,
    DEFAULT_CLASS_NAMES,
    EXACT_INVALID_INPUT_MESSAGE,
)
from src.ood_detector import OODGatekeeper

compute_energy_score = OODGatekeeper.compute_energy_score

__all__ = [
    "predict_single_image",
    "compute_energy_score",
    "DEFAULT_CLASS_NAMES",
    "EXACT_INVALID_INPUT_MESSAGE",
]

if __name__ == "__main__":
    import argparse
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
