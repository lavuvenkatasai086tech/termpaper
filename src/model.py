import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from transformers import ViTModel, ViTConfig


class SEGatedFeatureFusion(nn.Module):
    """
    Squeeze-and-Excitation (SE) Gated Feature Fusion module.
    Dynamically weights local CNN features (EfficientNetV2-B3) vs. global contextual
    features (ViT-Base) to resolve confusion between localized lesion patterns (Pest)
    and broader systemic foliar chlorosis (Fungi / Blight), as suggested in Sinamenye et al. (2025).

    The module produces calibrated modality weights (w_cnn, w_vit) such that:
        w_cnn + w_vit = 1.0 (softmax normalized)
    and exposes live percentage splits (CNN % vs. ViT %) for diagnostic interpretability.
    """
    def __init__(self, cnn_dim: int = 1536, vit_dim: int = 512, reduction: int = 8):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.vit_dim = vit_dim
        total_dim = cnn_dim + vit_dim  # 2048

        # Squeeze-and-Excitation bottleneck for cross-modal gating
        bottleneck_dim = max(32, total_dim // reduction)
        self.gate_fc1 = nn.Linear(total_dim, bottleneck_dim, bias=False)
        self.gate_act = nn.ReLU(inplace=True)
        self.gate_fc2 = nn.Linear(bottleneck_dim, 2, bias=True)

        # Zero-initialization: starts with equal 50% / 50% neutral weights
        # guaranteeing 100% backward compatibility with pre-trained checkpoints
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor):
        """
        Args:
            cnn_feat: Local CNN representations [B, 1536]
            vit_feat: Global ViT representations [B, 512]
        Returns:
            fused: Modality-weighted concatenated vector [B, 2048]
            modality_weights: Tensor of shape [B, 2] containing (w_cnn, w_vit)
        """
        raw_concat = torch.cat([cnn_feat, vit_feat], dim=-1)  # [B, 2048]
        logits = self.gate_fc2(self.gate_act(self.gate_fc1(raw_concat)))  # [B, 2]
        modality_weights = F.softmax(logits, dim=-1)  # [B, 2]

        w_cnn = modality_weights[:, 0:1]  # [B, 1]
        w_vit = modality_weights[:, 1:2]  # [B, 1]

        # Scale features dynamically by 2.0 * weight so neutral 0.5, 0.5 retains exact 1.0 scale
        scaled_cnn = 2.0 * w_cnn * cnn_feat
        scaled_vit = 2.0 * w_vit * vit_feat
        fused = torch.cat([scaled_cnn, scaled_vit], dim=-1)  # [B, 2048]

        return fused, modality_weights


class EnhancedHybridEfficientNetViT(nn.Module):
    """
    Enhanced Hybrid Deep Learning Architecture combining EfficientNetV2-B3 and Vision Transformer (ViT)
    with SE-Gated Feature Fusion and configurable partial backbone fine-tuning.
    Based on:
    'Potato plant disease detection: leveraging hybrid deep learning models' (Sinamenye et al., BMC Plant Biology 2025).

    Enhancements:
    1. Squeeze-and-Excitation (SE) Gated Feature Fusion module between local CNN and global ViT paths.
    2. Exposes live modality weight split (CNN % vs. ViT %) in forward pass and inference.
    3. Partial unfreezing support for the top stage of EfficientNetV2-B3 and top 2 blocks of ViT-Base
       to eliminate the 20% Pest vs. Fungi confusion reported in the paper.
    4. Full support for Grad-CAM (target layer conv_head) and ViT Attention Rollout.
    """
    def __init__(
        self,
        num_classes: int = 7,
        effnet_model_name: str = "tf_efficientnetv2_b3.in21k",
        vit_model_name: str = "google/vit-base-patch16-224-in21k",
        freeze_backbone: bool = True,
        unfreeze_top_blocks: bool = False,
        use_se_gating: bool = True,
        dropout_rate: float = 0.2,
        vit_embed_dim: int = 512,
        weights_dir: str = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_se_gating = use_se_gating

        if weights_dir is None:
            weights_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")

        # ---------------------------------------------------------
        # Path 1: EfficientNetV2B3 for compact local visual features
        # ---------------------------------------------------------
        local_effnet_path = os.path.join(weights_dir, "effnetv2b3_in21k.safetensors")
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

        # Remove default classification head to expose GAP features
        self.effnet.classifier = nn.Identity()

        if freeze_backbone:
            for param in self.effnet.parameters():
                param.requires_grad = False

        self.effnet_dim = getattr(self.effnet, "num_features", 1536)

        # ---------------------------------------------------------
        # Path 2: Vision Transformer for global contextual features
        # ---------------------------------------------------------
        local_vit_weights = os.path.join(weights_dir, "vit_base_patch16_224_in21k.safetensors")
        local_vit_config = os.path.join(weights_dir, "vit_config.json")

        if os.path.exists(local_vit_weights) and os.path.exists(local_vit_config):
            try:
                import safetensors.torch
                vit_cfg = ViTConfig.from_json_file(local_vit_config)
                # Eager attention implementation ensures output_attentions works cleanly
                vit_cfg._attn_implementation = "eager"
                self.vit = ViTModel(vit_cfg)
                vit_state = safetensors.torch.load_file(local_vit_weights)
                self.vit.load_state_dict(vit_state, strict=False)
            except Exception as e:
                print(f"[Model Warning] Could not load local ViT safetensors: {e}, falling back to pretrained loader")
                self.vit = ViTModel.from_pretrained(vit_model_name, attn_implementation="eager")
        else:
            self.vit = ViTModel.from_pretrained(vit_model_name, attn_implementation="eager")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.vit_hidden_size = self.vit.config.hidden_size  # 768
        self.vit_linear = nn.Linear(self.vit_hidden_size, vit_embed_dim)

        # ---------------------------------------------------------
        # Path 3: Squeeze-and-Excitation Gated Feature Fusion
        # ---------------------------------------------------------
        self.se_gate = SEGatedFeatureFusion(
            cnn_dim=self.effnet_dim,
            vit_dim=vit_embed_dim,
            reduction=8,
        )

        # ---------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fusion_dim = self.effnet_dim + vit_embed_dim  # 1536 + 512 = 2048
        self.classifier = nn.Linear(self.fusion_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

        # Optional partial unfreezing of top blocks
        if unfreeze_top_blocks:
            self.configure_partial_unfreezing(True)

    def configure_partial_unfreezing(self, unfreeze_top_blocks: bool = True):
        """
        Unfreezes the final stage of EfficientNetV2-B3 and top 2 blocks of ViT-Base
        to enable fine-grained adaptation for confusing class boundaries (Pest vs. Fungi).
        """
        if not unfreeze_top_blocks:
            return

        # 1. Unfreeze top stage of EfficientNetV2-B3
        if hasattr(self.effnet, "conv_head"):
            for param in self.effnet.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.effnet, "bn2"):
            for param in self.effnet.bn2.parameters():
                param.requires_grad = True
        if hasattr(self.effnet, "blocks") and len(self.effnet.blocks) > 0:
            for param in self.effnet.blocks[-1].parameters():
                param.requires_grad = True

        # 2. Unfreeze top 2 transformer encoder layers of ViT-Base
        if hasattr(self.vit, "encoder") and hasattr(self.vit.encoder, "layer"):
            for layer in self.vit.encoder.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def get_parameter_groups(self, backbone_lr: float = 1e-5, head_lr: float = 1e-4, weight_decay: float = 1e-5):
        """
        Returns differential learning rate parameter groups:
        lower lr (1e-5) for pre-trained backbone blocks, higher lr (1e-4) for SE gate and classification head.
        """
        backbone_params = []
        head_params = []

        # EffNet trainable params
        for p in self.effnet.parameters():
            if p.requires_grad:
                backbone_params.append(p)

        # ViT trainable params
        for p in self.vit.parameters():
            if p.requires_grad:
                backbone_params.append(p)

        # Head and gate params
        for p in self.vit_linear.parameters():
            if p.requires_grad:
                head_params.append(p)
        for p in self.se_gate.parameters():
            if p.requires_grad:
                head_params.append(p)
        for p in self.classifier.parameters():
            if p.requires_grad:
                head_params.append(p)

        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

    def extract_local_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts local convolutional representations via GAP: [B, 1536].
        """
        eff_features = self.effnet.forward_features(x)
        if eff_features.dim() == 4:
            eff_features = torch.flatten(eff_features, start_dim=2).mean(dim=2)
        elif eff_features.dim() == 3:
            eff_features = eff_features.mean(dim=1)
        return eff_features

    def extract_global_features(self, x: torch.Tensor, output_attentions: bool = False):
        """
        Extracts global transformer representations via CLS token and projection: [B, 512].
        Optionally returns raw multi-head self-attention maps across all 12 layers.
        """
        vit_out = self.vit(pixel_values=x, output_attentions=output_attentions)
        cls_token = vit_out.last_hidden_state[:, 0]
        global_features = self.vit_linear(cls_token)
        if output_attentions:
            return global_features, vit_out.attentions
        return global_features

    def extract_features(self, x: torch.Tensor):
        """
        Returns individual and fused feature vectors:
        (fused_features [B, 2048], cnn_features [B, 1536], vit_features [B, 512], modality_weights [B, 2])
        """
        local_vec = self.extract_local_features(x)
        global_vec = self.extract_global_features(x)
        if self.use_se_gating:
            fused, modality_weights = self.se_gate(local_vec, global_vec)
        else:
            fused = torch.cat([local_vec, global_vec], dim=-1)
            b = x.size(0)
            modality_weights = torch.full((b, 2), 0.5, device=x.device)
        return fused, local_vec, global_vec, modality_weights

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
        return_modality_weights: bool = False,
    ):
        """
        Forward pass through parallel paths, SE-gated fusion, and classification head.

        Args:
            x (torch.Tensor): Input batch [B, 3, 224, 224]
            return_probs (bool): If True, returns class probabilities via Softmax
            return_modality_weights (bool): If True, returns (output, modality_weights)
        """
        fused, _, _, modality_weights = self.extract_features(x)
        fused_reg = self.dropout(fused)
        logits = self.classifier(fused_reg)

        output = self.softmax(logits) if return_probs else logits

        if return_modality_weights:
            return output, modality_weights
        return output

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates output class probabilities applying Softmax activation.
        """
        return self.forward(x, return_probs=True)

    @torch.no_grad()
    def get_modality_weights(self, x: torch.Tensor):
        """
        Returns live modality percentage split for an input batch:
        {"cnn_pct": float, "vit_pct": float}
        """
        _, _, _, weights = self.extract_features(x)
        weights_np = weights.cpu().numpy()
        cnn_pct = float(weights_np[0, 0] * 100.0)
        vit_pct = float(weights_np[0, 1] * 100.0)
        return {"cnn_pct": round(cnn_pct, 2), "vit_pct": round(vit_pct, 2)}


# Subclass / Alias for 100% backward compatibility
class EfficientNetV2B3ViT(EnhancedHybridEfficientNetViT):
    pass
