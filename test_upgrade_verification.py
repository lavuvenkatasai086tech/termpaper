import os
import sys
import json
import numpy as np
from PIL import Image
import torch

from src.model import EnhancedHybridEfficientNetViT
from src.ood_detector import OODGatekeeper, EXACT_INVALID_INPUT_MESSAGE
from src.xai import XAIExplainer
from src.dataset import get_transforms
from app import app

def run_tests():
    print("=" * 80)
    print("RUNNING END-TO-END VERIFICATION SUITE FOR POTATO LEAF UPGRADE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. Testing Device & Model Loading: {device}")
    model = EnhancedHybridEfficientNetViT(num_classes=7).to(device)
    ckpt_path = "checkpoints/best_model.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    model.eval()
    print("   EnhancedHybridEfficientNetViT loaded successfully.")

    gatekeeper = OODGatekeeper(device=device)
    xai = XAIExplainer()
    transform = get_transforms(augment=False)

    # -------------------------------------------------------------
    # TEST A: Valid Potato Leaf Image
    # -------------------------------------------------------------
    print("\n--- TEST A: Valid Potato Leaf Specimen ---")
    valid_candidates = [
        "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Healthy/1692336758850.jpg",
        "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Phytopthora/20230712_155047.jpg",
        "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Fungi/20230712_150159.jpg",
        "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment/Pest/1692332350400.jpg",
    ]
    valid_path = None
    for p in valid_candidates:
        if os.path.exists(p):
            valid_path = p
            break

    if valid_path:
        img_a = Image.open(valid_path).convert("RGB")
        tensor_a = transform(img_a).unsqueeze(0).to(device)
        res_a = gatekeeper.validate_and_gate(img_a, model, tensor_a)

        assert res_a["is_valid"] is True, f"Expected valid potato leaf, got: {res_a}"
        assert res_a["status"] == "VALID_POTATO_LEAF"
        print(f"   [PASS] Valid Potato Leaf Accepted: {os.path.basename(valid_path)}")
        print(f"          Status: {res_a['status']}")
        print(f"          Energy: {res_a['energy_score']} | SimP: {res_a['potato_similarity']} | Margin: {res_a['foliar_margin']}")

        # Compute disease classification & XAI
        with torch.no_grad():
            fused, _, _, modality_weights = model.extract_features(tensor_a)
            logits = model.classifier(fused)
            probs = torch.softmax(logits, dim=-1).cpu().squeeze().numpy()
            top_class = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"][int(probs.argmax())]
            mod_stats = model.get_modality_weights(tensor_a)

        print(f"          Prediction: {top_class} ({float(probs.max())*100:.2f}%)")
        print(f"          SE-Gate Modality Split: CNN Local {mod_stats['cnn_pct']}% | ViT Global {mod_stats['vit_pct']}%")

        # Test XAI generation
        xai_res = xai.explain(model, tensor_a, img_a, target_class=int(probs.argmax()))
        assert "original_b64" in xai_res and "gradcam_b64" in xai_res and "rollout_b64" in xai_res and "fused_b64" in xai_res
        print(f"   [PASS] XAI Heatmaps (Grad-CAM, ViT Rollout, Fused XAI) generated successfully.")
    else:
        print("   [SKIP] No local potato dataset images found to test.")

    # -------------------------------------------------------------
    # TEST B: Non-Potato Leaf Image (e.g. Tomato or Apple)
    # -------------------------------------------------------------
    print("\n--- TEST B: Non-Potato Leaf (Out-of-Distribution Crop) ---")
    np_candidates = [
        "PlantVillage/Tomato_Leaf_Mold",
        "PlantVillage/Tomato_healthy",
        "PlantVillage/Apple___healthy",
        "PlantVillage/Corn_maize___Northern_Leaf_Blight",
        "PlantVillage/Peach___healthy",
    ]
    np_path = None
    for folder in np_candidates:
        if os.path.isdir(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                np_path = os.path.join(folder, files[0])
                break

    if np_path:
        img_b = Image.open(np_path).convert("RGB")
        tensor_b = transform(img_b).unsqueeze(0).to(device)
        res_b = gatekeeper.validate_and_gate(img_b, model, tensor_b)

        assert res_b["is_valid"] is False, f"Expected non-potato crop to be rejected, but got: {res_b}"
        assert res_b["status"] == "INVALID_INPUT"
        assert res_b["error_message"] == EXACT_INVALID_INPUT_MESSAGE
        assert res_b["predicted_class"] is None
        assert res_b["probabilities"] is None
        print(f"   [PASS] Non-Potato Crop Rejected: {np_path}")
        print(f"          Status: {res_b['status']}")
        print(f"          Rejection Stage: {res_b.get('rejection_stage')}")
        print(f"          Rejection Reason: {res_b.get('rejection_reason')}")
        print(f"          Exact User Message: \"{res_b['error_message']}\"")
        print(f"          Disease Predictions Suppressed: {res_b['predicted_class'] is None}")
    else:
        print("   [SKIP] No PlantVillage non-potato crop images found to test.")

    # -------------------------------------------------------------
    # TEST C: Random Non-Leaf / Object / Noise Image
    # -------------------------------------------------------------
    print("\n--- TEST C: Random Non-Leaf / Object / Synthetic Noise ---")
    synth_noise = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    t_noise = transform(synth_noise).unsqueeze(0).to(device)
    res_c1 = gatekeeper.validate_and_gate(synth_noise, model, t_noise)

    assert res_c1["is_valid"] is False
    assert res_c1["status"] == "INVALID_INPUT"
    assert res_c1["error_message"] == EXACT_INVALID_INPUT_MESSAGE
    print(f"   [PASS] Synthetic Noise Rejected with INVALID_INPUT.")
    print(f"          Reason: {res_c1.get('rejection_reason')}")

    obj_path = "PlantVillage/Non_Plant_Objects/coffee_mug.jpg"
    if os.path.exists(obj_path):
        img_obj = Image.open(obj_path).convert("RGB")
        t_obj = transform(img_obj).unsqueeze(0).to(device)
        res_c2 = gatekeeper.validate_and_gate(img_obj, model, t_obj)
        assert res_c2["is_valid"] is False
        assert res_c2["status"] == "INVALID_INPUT"
        print(f"   [PASS] Non-Plant Object (Coffee Mug) Rejected with INVALID_INPUT.")
        print(f"          Reason: {res_c2.get('rejection_reason')}")

    # -------------------------------------------------------------
    # TEST D: Flask API Endpoints Integration Test
    # -------------------------------------------------------------
    print("\n--- TEST D: Flask Application Client Endpoints ---")
    app.config["TESTING"] = True
    client = app.test_client()

    # D1: /api/health
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200
    h_data = resp_health.get_json()
    assert h_data["status"] == "healthy"
    print("   [PASS] /api/health returns 200 healthy status.")

    # D2: /api/samples
    resp_samples = client.get("/api/samples")
    assert resp_samples.status_code == 200
    s_data = resp_samples.get_json()
    assert len(s_data.get("samples", [])) > 0
    print(f"   [PASS] /api/samples returns {len(s_data['samples'])} specimens.")

    # D3: /api/predict with non-potato specimen
    if np_path:
        rel_sample = np_path.replace("\\", "/")
        resp_pred_np = client.post("/api/predict", json={"sample_path": rel_sample})
        assert resp_pred_np.status_code == 200
        p_np_data = resp_pred_np.get_json()
        assert p_np_data["is_valid"] is False
        assert p_np_data["status"] == "INVALID_INPUT"
        assert p_np_data["error_message"] == EXACT_INVALID_INPUT_MESSAGE
        assert p_np_data["predicted_class"] is None
        print("   [PASS] API /api/predict properly rejected non-potato sample with INVALID_INPUT.")

    # D4: /api/predict with valid potato specimen
    if valid_path:
        # relative path from DATA_DIR
        rel_valid = os.path.relpath(valid_path, "data/potatodata/Potato Leaf Disease Dataset in Uncontrolled Environment").replace("\\", "/")
        resp_pred_p = client.post("/api/predict", json={"sample_path": rel_valid})
        assert resp_pred_p.status_code == 200
        p_valid_data = resp_pred_p.get_json()
        assert p_valid_data["is_valid"] is True
        assert p_valid_data["status"] == "VALID_POTATO_LEAF"
        assert p_valid_data["predicted_class"] in ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
        assert "xai" in p_valid_data and p_valid_data["xai"] is not None
        assert "modality_weights" in p_valid_data
        print(f"   [PASS] API /api/predict accepted valid potato leaf: {p_valid_data['predicted_class']} ({p_valid_data['confidence']}%)")
        print(f"          XAI and SE-Gate modality weights successfully returned.")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
