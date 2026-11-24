import io
import base64
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F


class XAIExplainer:
    """
    Explainable AI (XAI) Suite for Hybrid EfficientNetV2-B3 + ViT-Base Architecture
    Based on Future Work in Sinamenye et al. (BMC Plant Biology, 2025).

    Components:
    1. CNN Grad-CAM: Gradient-weighted Class Activation Mapping on the final
       convolutional block of EfficientNetV2-B3 (local lesions, fungal spots, CPB chews).
    2. ViT Attention Rollout: Recursive multi-head self-attention rollout across
       all 12 transformer layers in ViT-Base (global leaf canopy, chlorosis, mosaic wilting).
    3. Fused XAI Overlay: Dynamically blended saliency weighted by the SE-Gate
       modality weights, rendered with interactive colormap overlays.
    """
    def __init__(self, colormap: int = cv2.COLORMAP_JET):
        self.colormap = colormap

    def compute_cnn_gradcam(
        self,
        model,
        tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Computes Grad-CAM heatmap on the final convolutional layer (conv_head) of EfficientNetV2-B3.
        Returns a 2D float32 numpy array [H, W] normalized between 0.0 and 1.0.
        """
        model.eval()

        # Locate target convolutional layer (conv_head or last conv block)
        target_layer = None
        if hasattr(model.effnet, "conv_head"):
            target_layer = model.effnet.conv_head
        elif hasattr(model.effnet, "bn2"):
            target_layer = model.effnet.bn2
        elif hasattr(model.effnet, "blocks") and len(model.effnet.blocks) > 0:
            target_layer = model.effnet.blocks[-1]

        if target_layer is None:
            # Fallback uniform heatmap if layer not found
            return np.ones((tensor.size(2), tensor.size(3)), dtype=np.float32) * 0.5

        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Enable grad on tensor and model parameters temporarily
            tensor_req = tensor.clone().detach().requires_grad_(True)
            
            # Forward pass through model
            fused, _, _, _ = model.extract_features(tensor_req)
            logits = model.classifier(fused)

            if target_class is None:
                target_class = int(logits.argmax(dim=-1).item())

            score = logits[0, target_class]
            model.zero_grad()
            score.backward(retain_graph=False)

            if not activations or not gradients:
                return np.ones((tensor.size(2), tensor.size(3)), dtype=np.float32) * 0.5

            act = activations[0].detach()  # [1, C, H, W]
            grad = gradients[0].detach()   # [1, C, H, W]

            # Pool gradients over spatial dimensions: alpha_k = GAP(grad)
            alpha = grad.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

            # Weighted combination of activation maps
            cam = (alpha * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
            cam = F.relu(cam)

            # Resize to input dimensions (224x224)
            cam = F.interpolate(cam, size=(tensor.size(2), tensor.size(3)), mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            c_min = float(cam_np.min())
            c_max = float(cam_np.max())
            if c_max > c_min:
                cam_np = (cam_np - c_min) / (c_max - c_min + 1e-8)
            else:
                cam_np = np.zeros_like(cam_np)

            return cam_np.astype(np.float32)

        finally:
            h_fwd.remove()
            h_bwd.remove()

    def compute_vit_attention_rollout(
        self,
        model,
        tensor: torch.Tensor,
        discard_ratio: float = 0.1,
        head_fusion: str = "mean",
    ) -> np.ndarray:
        """
        Computes ViT Attention Rollout across all 12 transformer encoder layers.
        Recursively rolls out multi-head self-attention from [CLS] to patch tokens.
        Returns a 2D float32 numpy array [H, W] normalized between 0.0 and 1.0.
        """
        model.eval()
        with torch.no_grad():
            # Ensure eager attention mode for attention map extraction
            if hasattr(model.vit, "config") and getattr(model.vit.config, "_attn_implementation", None) != "eager":
                model.vit.config._attn_implementation = "eager"

            vit_out = model.vit(pixel_values=tensor, output_attentions=True)
            attentions = vit_out.attentions  # tuple of 12 tensors: [1, num_heads, 197, 197]

            if not attentions or len(attentions) == 0:
                return np.ones((tensor.size(2), tensor.size(3)), dtype=np.float32) * 0.5

            num_tokens = attentions[0].size(-1)  # 197 = 1 CLS + 196 patches (14x14)
            result = torch.eye(num_tokens, device=tensor.device)

            for att in attentions:
                # att: [1, num_heads, 197, 197]
                if head_fusion == "mean":
                    att_fused = att.mean(dim=1).squeeze(0)  # [197, 197]
                elif head_fusion == "max":
                    att_fused = att.max(dim=1).values.squeeze(0)
                else:
                    att_fused = att.min(dim=1).values.squeeze(0)

                # Optional low-attention noise discard
                if discard_ratio > 0.0:
                    flat = att_fused.view(-1)
                    k = int(flat.size(0) * discard_ratio)
                    if k > 0:
                        thresh = torch.kthvalue(flat, k).values
                        att_fused = torch.where(att_fused < thresh, torch.zeros_like(att_fused), att_fused)

                # Add residual identity connection and row-normalize
                i_eye = torch.eye(num_tokens, device=tensor.device)
                a_hat = 0.5 * att_fused + 0.5 * i_eye
                row_sum = a_hat.sum(dim=-1, keepdim=True) + 1e-8
                a_hat = a_hat / row_sum

                result = torch.matmul(a_hat, result)

            # Attention from CLS token (token 0) to all 196 spatial patches (tokens 1..196)
            cls_attention = result[0, 1:]  # [196]
            grid_size = int(np.sqrt(cls_attention.size(0)))  # 14
            mask = cls_attention.reshape(1, 1, grid_size, grid_size)

            # Bilinear upsampling to 224x224
            mask_upsampled = F.interpolate(
                mask,
                size=(tensor.size(2), tensor.size(3)),
                mode="bilinear",
                align_corners=False
            ).squeeze().cpu().numpy()

            # Normalize to [0, 1]
            m_min = float(mask_upsampled.min())
            m_max = float(mask_upsampled.max())
            if m_max > m_min:
                mask_norm = (mask_upsampled - m_min) / (m_max - m_min + 1e-8)
            else:
                mask_norm = np.zeros_like(mask_upsampled)

            return mask_norm.astype(np.float32)

    def apply_colormap_and_blend(
        self,
        rgb_img_np: np.ndarray,
        heatmap_norm: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Applies JET colormap to normalized heatmap and blends with original RGB image:
        blended = (1 - alpha) * RGB + alpha * Heatmap
        """
        # Ensure RGB is uint8 [H, W, 3]
        if rgb_img_np.dtype != np.uint8:
            rgb_img_np = np.clip(rgb_img_np * 255.0, 0, 255).astype(np.uint8)

        # Heatmap to uint8 [H, W]
        h_uint8 = np.clip(heatmap_norm * 255.0, 0, 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(h_uint8, self.colormap)
        # cv2 outputs BGR, convert to RGB
        color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        # Blend
        blended = (1.0 - alpha) * rgb_img_np.astype(np.float32) + alpha * color_heatmap_rgb.astype(np.float32)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return blended

    @staticmethod
    def numpy_to_base64_png(img_np: np.ndarray) -> str:
        """
        Converts uint8 RGB image array to base64-encoded PNG data URL.
        """
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def explain(
        self,
        model,
        tensor: torch.Tensor,
        image_pil: Image.Image,
        target_class: int = None,
        alpha: float = 0.5,
    ) -> dict:
        """
        Generates full explainability suite:
        - Original RGB preview
        - CNN Grad-CAM (local lesion focus)
        - ViT Attention Rollout (global canopy context)
        - Fused XAI Overlay (modality-weighted blend)
        - SE-Gate Modality Weights (CNN % vs ViT %)

        Returns dictionary of base64 PNG images and modality stats.
        """
        # Resize original image to 224x224 for pixel-perfect alignment
        img_rgb = image_pil.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        img_rgb_np = np.array(img_rgb)

        # 1. CNN Grad-CAM
        gradcam_map = self.compute_cnn_gradcam(model, tensor, target_class=target_class)

        # 2. ViT Attention Rollout
        rollout_map = self.compute_vit_attention_rollout(model, tensor)

        # 3. SE-Gate Modality Weights
        with torch.no_grad():
            modality_stats = model.get_modality_weights(tensor)
        w_cnn = modality_stats["cnn_pct"] / 100.0
        w_vit = modality_stats["vit_pct"] / 100.0

        # 4. Fused Heatmap
        fused_map = w_cnn * gradcam_map + w_vit * rollout_map
        f_min, f_max = float(fused_map.min()), float(fused_map.max())
        if f_max > f_min:
            fused_map = (fused_map - f_min) / (f_max - f_min + 1e-8)
        else:
            fused_map = np.zeros_like(fused_map)

        # 5. Colormap Blends
        gradcam_blend = self.apply_colormap_and_blend(img_rgb_np, gradcam_map, alpha=alpha)
        rollout_blend = self.apply_colormap_and_blend(img_rgb_np, rollout_map, alpha=alpha)
        fused_blend = self.apply_colormap_and_blend(img_rgb_np, fused_map, alpha=alpha)

        return {
            "original_b64": self.numpy_to_base64_png(img_rgb_np),
            "gradcam_b64": self.numpy_to_base64_png(gradcam_blend),
            "rollout_b64": self.numpy_to_base64_png(rollout_blend),
            "fused_b64": self.numpy_to_base64_png(fused_blend),
            "modality_weights": {
                "cnn_pct": modality_stats["cnn_pct"],
                "vit_pct": modality_stats["vit_pct"]
            },
            "alpha": alpha,
            "target_class": target_class,
        }
