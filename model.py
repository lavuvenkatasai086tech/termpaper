"""
Root compatibility wrapper for model.py.
Imports enhanced models from src.model.
"""
from src.model import (
    EnhancedHybridEfficientNetViT,
    EfficientNetV2B3ViT,
    SEGatedFeatureFusion,
)

__all__ = [
    "EnhancedHybridEfficientNetViT",
    "EfficientNetV2B3ViT",
    "SEGatedFeatureFusion",
]

if __name__ == "__main__":
    import torch
    print("[Testing model.py compatibility wrapper]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedHybridEfficientNetViT(num_classes=7).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    logits, weights = model(dummy_input, return_modality_weights=True)
    probs = model.predict_proba(dummy_input)
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Modality Weights: {weights.cpu().numpy()}")
    print(f"Output Probs Shape:  {probs.shape}")
