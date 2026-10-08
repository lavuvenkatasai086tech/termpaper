import os
import torch
import torch.nn as nn
from timm import create_model
from transformers import ViTModel


class EfficientNetV2B3ViT(nn.Module):
    """
    Hybrid Deep Learning Architecture combining EfficientNetV2B3 and Vision Transformer (ViT)
    as presented in:
    'Potato plant disease detection: leveraging hybrid deep learning models' (Sinamenye et al., 2025).

    Architecture details:
    1. Dual Feature Extraction Paths:
       - Path 1 (CNN - Local Features):
           * Pre-trained EfficientNetV2B3 backbone (weights frozen).
           * Classifier removed.
           * Feature map flattened across spatial dimensions and processed via Global Average Pooling (GAP)
             to extract a 1536-dimensional local feature vector.
       - Path 2 (ViT - Global Contextual Features):
           * Pre-trained Vision Transformer Base (google/vit-base-patch16-224-in21k, patch size 16x16, input 224x224).
           * Parameters frozen.
           * CLS token extracted from last hidden state (768-dim) and passed through a Linear projection
             layer to reduce dimensionality to 512 dimensions.
    2. Feature Fusion & Classification Head:
       - Local (1536) and Global (512) feature vectors concatenated (total dimension = 2048).
       - Dropout layer (rate = 0.2).
       - Fully Connected Dense layer mapping 2048 -> num_classes (default: 7).
       - Returns raw logits for numerically stable CrossEntropyLoss, with a softmax method for probability inference.
    """
    def __init__(
        self,
        num_classes: int = 7,
        effnet_model_name: str = "tf_efficientnetv2_b3.in21k",
        vit_model_name: str = "google/vit-base-patch16-224-in21k",
        freeze_backbone: bool = True,
        dropout_rate: float = 0.2,
        vit_embed_dim: int = 512,
    ):
        super(EfficientNetV2B3ViT, self).__init__()
        self.num_classes = num_classes

        # ---------------------------------------------------------
        # Path 1: EfficientNetV2B3 for compact local visual features
        # ---------------------------------------------------------
        local_effnet_path = os.path.join(os.path.dirname(__file__), "weights", "effnetv2b3_in21k.safetensors")
        if os.path.exists(local_effnet_path):
            self.effnet = create_model(effnet_model_name, pretrained=False)
            try:
                import safetensors.torch
                eff_state = safetensors.torch.load_file(local_effnet_path)
                self.effnet.load_state_dict(eff_state, strict=False)
            except Exception as e:
                print(f"[Model Warning] Could not load local effnet safetensors: {e}")
        else:
            self.effnet = create_model(effnet_model_name, pretrained=True)

        # Remove default classification head
        self.effnet.classifier = nn.Identity()

        if freeze_backbone:
            for param in self.effnet.parameters():
                param.requires_grad = False

        # Determine EfficientNet output feature dimension (typically 1536 for B3)
        self.effnet_dim = getattr(self.effnet, "num_features", 1536)

        # ---------------------------------------------------------
        # Path 2: Vision Transformer for global contextual features
        # ---------------------------------------------------------
        local_vit_weights = os.path.join(os.path.dirname(__file__), "weights", "vit_base_patch16_224_in21k.safetensors")
        local_vit_config = os.path.join(os.path.dirname(__file__), "weights", "vit_config.json")

        if os.path.exists(local_vit_weights) and os.path.exists(local_vit_config):
            try:
                import safetensors.torch
                from transformers import ViTConfig
                vit_cfg = ViTConfig.from_json_file(local_vit_config)
                self.vit = ViTModel(vit_cfg)
                vit_state = safetensors.torch.load_file(local_vit_weights)
                self.vit.load_state_dict(vit_state, strict=False)
            except Exception as e:
                print(f"[Model Warning] Could not load local ViT safetensors: {e}, falling back to pretrained loader")
                self.vit = ViTModel.from_pretrained(vit_model_name)
        else:
            self.vit = ViTModel.from_pretrained(vit_model_name)

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.vit_hidden_size = self.vit.config.hidden_size  # 768
        self.vit_linear = nn.Linear(self.vit_hidden_size, vit_embed_dim)

        # ---------------------------------------------------------
        # Feature Fusion & Classification Head
        # ---------------------------------------------------------
        self.dropout = nn.Dropout(p=dropout_rate)
        fusion_dim = self.effnet_dim + vit_embed_dim  # 1536 + 512 = 2048
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def extract_local_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts local convolutional representations via GAP.
        """
        # Feature map shape: [Batch, Channels, H, W]
        eff_features = self.effnet.forward_features(x)
        # Flatten spatial dims [Batch, Channels, H*W] and GAP over spatial locations
        if eff_features.dim() == 4:
            eff_features = torch.flatten(eff_features, start_dim=2).mean(dim=2)
        elif eff_features.dim() == 3:
            eff_features = eff_features.mean(dim=1)
        return eff_features

    def extract_global_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts global transformer representations via CLS token and projection.
        """
        vit_out = self.vit(pixel_values=x)
        # Extract [CLS] token at index 0: [Batch, 768]
        cls_token = vit_out.last_hidden_state[:, 0]
        # Align/reduce dimensionality to 512
        global_features = self.vit_linear(cls_token)
        return global_features

    def forward(self, x: torch.Tensor, return_probs: bool = False) -> torch.Tensor:
        """
        Forward pass through parallel paths, fusion, and classification.

        Args:
            x (torch.Tensor): Input batch of normalized RGB images [B, 3, 224, 224]
            return_probs (bool): If True, returns class probabilities via Softmax;
                                 else returns raw logits for CrossEntropyLoss.
        """
        local_vec = self.extract_local_features(x)    # [B, 1536]
        global_vec = self.extract_global_features(x)  # [B, 512]

        # Fusion via concatenation: [B, 2048]
        fused = torch.cat((local_vec, global_vec), dim=1)

        # Regularization
        fused = self.dropout(fused)

        # Linear classification
        logits = self.classifier(fused)

        if return_probs:
            return self.softmax(logits)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates output class probabilities applying Softmax activation.
        """
        logits = self.forward(x, return_probs=False)
        return self.softmax(logits)


if __name__ == "__main__":
    print("[Testing model.py]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = EfficientNetV2B3ViT(num_classes=7).to(device)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    logits = model(dummy_input)
    probs = model.predict_proba(dummy_input)
    print(f"Output Logits Shape: {logits.shape}")
    print(f"Output Probs Shape:  {probs.shape}")
    print(f"Probs Sum: {probs.sum(dim=1)}")
