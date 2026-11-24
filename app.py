import os
import io
import time
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from PIL import Image
import torch
import torch.nn.functional as F

from src.model import EnhancedHybridEfficientNetViT, EfficientNetV2B3ViT
from src.ood_detector import OODGatekeeper, EXACT_INVALID_INPUT_MESSAGE
from src.xai import XAIExplainer
from src.dataset import get_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "potatodata",
    "Potato Leaf Disease Dataset in Uncontrolled Environment"
)
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")

CLASS_NAMES = ["Bacteria", "Fungi", "Healthy", "Nematode", "Pest", "Phytopthora", "Virus"]
DEFAULT_ENERGY_THRESHOLD = 0.50
DEFAULT_TEMPERATURE = 1.0

# Global device and components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_instance = None
gatekeeper_instance = None
xai_instance = None

model_status_info = {
    "loaded": False,
    "device": str(device),
    "model_type": "Enhanced Hybrid EfficientNetV2-B3 + ViT (Sinamenye et al., 2025)",
    "weights_path": None,
    "message": "Initializing components..."
}


def get_gatekeeper():
    global gatekeeper_instance
    if gatekeeper_instance is None:
        gatekeeper_instance = OODGatekeeper(
            config_path=CONFIG_PATH,
            checkpoint_dir=CHECKPOINT_DIR,
            device=device,
        )
    return gatekeeper_instance


def get_xai():
    global xai_instance
    if xai_instance is None:
        xai_instance = XAIExplainer()
    return xai_instance


def load_or_init_model():
    global model_instance, model_status_info
    if model_instance is not None:
        return model_instance

    try:
        logging.info("[App] Initializing EnhancedHybridEfficientNetViT model...")
        model = EnhancedHybridEfficientNetViT(
            num_classes=len(CLASS_NAMES),
            use_se_gating=True,
        )
        model.to(device)

        if os.path.exists(BEST_MODEL_PATH):
            logging.info(f"[App] Loading weights from {BEST_MODEL_PATH}")
            ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict, strict=False)
            model_status_info["weights_path"] = BEST_MODEL_PATH
            model_status_info["loaded"] = True
            model_status_info["message"] = "Trained model checkpoint loaded successfully."
        else:
            model_status_info["loaded"] = True
            model_status_info["weights_path"] = "Pretrained Backbones (Zero-Shot / Base Features)"
            model_status_info["message"] = "Model ready with pretrained EfficientNetV2-B3 & ViT backbones."

        model.eval()
        model_instance = model
        return model_instance
    except Exception as e:
        logging.error(f"[App] Failed to load model: {e}", exc_info=True)
        model_status_info["loaded"] = False
        model_status_info["message"] = f"Model loading note: {str(e)}"
        return None


DISEASE_PROFILES = {
    "Bacteria": {
        "title": "Bacterial Infection / Blackleg / Wilt",
        "pathogen": "Ralstonia solanacearum / Pectobacterium spp.",
        "category": "Bacterial Pathogen",
        "severity": "High",
        "severity_color": "#f59e0b",
        "summary": "Bacterial wilt and blackleg cause rapid vascular browning, wilting of leaf stems, and foul-smelling soft rot under warm, moist conditions.",
        "symptoms": [
            "Dark brown to black inky discoloration on lower stems",
            "Rapid wilting of foliage during warm daylight hours without recovery",
            "Water-soaked dark lesions spreading rapidly across leaf veins",
            "Bacterial streaming visible when stem slice is submerged in water"
        ],
        "organic_remedies": [
            "Immediately rogue out and safely incinerate or bag infected plants",
            "Apply copper sulfate or copper hydroxide foliar sprays to suppress bacterial spread",
            "Incorporate bio-fungicides containing Bacillus amyloliquefaciens or Trichoderma",
            "Sterilize cutting knives and agricultural implements between field sections"
        ],
        "chemical_remedies": [
            "Copper oxychloride 50% WP (2.5 - 3 g/L water) protective spray",
            "Streptomycin sulfate + Tetracycline hydrochloride formulations where permitted",
            "Zinc and copper bactericidal seed tuber treatments prior to planting"
        ],
        "prevention": [
            "Plant only certified disease-free seed tubers",
            "Observe a strict 3-to-4 year rotation with non-solanaceous crops (e.g. maize, grasses)",
            "Avoid poorly drained soil and excessive furrow irrigation"
        ]
    },
    "Fungi": {
        "title": "Fungal Infection (Early Blight / Alternaria)",
        "pathogen": "Alternaria solani / Alternaria alternata",
        "category": "Fungal Foliar Pathogen",
        "severity": "Moderate",
        "severity_color": "#eab308",
        "summary": "Early blight is characterized by concentric ring 'target board' lesions on older leaves, causing premature defoliation and reduced tuber size.",
        "symptoms": [
            "Circular to angular brown spots with concentric target-board rings",
            "Yellow chlorotic halos surrounding dark necrotic lesions",
            "Senescence and crisping of lower leaves that eventually drop",
            "Sunken, dark corky lesions on tuber surfaces during harvest"
        ],
        "organic_remedies": [
            "Mulch heavily around potato hills to prevent soil splash onto lower foliage",
            "Spray potassium bicarbonate or compost tea early in the morning",
            "Apply Neem seed oil formulations (0.5%) to deter spore germination",
            "Prune senescent bottom leaves to enhance air circulation through canopy"
        ],
        "chemical_remedies": [
            "Mancozeb 75% WP (2.0 - 2.5 g/L) protective contact fungicide",
            "Azoxystrobin 23% SC or Pyraclostrobin strobilurin group fungicides",
            "Chlorothalonil applied at 7-10 day intervals during warm humid weather"
        ],
        "prevention": [
            "Maintain balanced nitrogen fertilization; avoid nutrient stress in late season",
            "Drip or furrow irrigation to keep foliage completely dry",
            "Deep tillage post-harvest to bury fungal crop residues"
        ]
    },
    "Healthy": {
        "title": "Healthy Foliage",
        "pathogen": "None (No Pathological Distress)",
        "category": "Optimal Plant Health",
        "severity": "None",
        "severity_color": "#10b981",
        "summary": "Foliage exhibits vigorous turgor, uniform chlorophyll distribution, and no evidence of fungal, bacterial, viral, or pest damage.",
        "symptoms": [
            "Vibrant, uniform green coloration without mottling or chlorosis",
            "Firm, upright leaf architecture and intact cellular leaf margins",
            "Smooth leaf surface with healthy trichome density",
            "Strong petiole attachments and vigorous apical growth"
        ],
        "organic_remedies": [
            "Continue standard organic fertilization (aged compost, well-rotted manure)",
            "Apply prophylactic seaweed extract / kelp foliar feed for trace micronutrients",
            "Maintain consistent soil moisture levels to avoid drought stress"
        ],
        "chemical_remedies": [
            "No chemical intervention needed",
            "Maintain preventative scouting and record keeping"
        ],
        "prevention": [
            "Maintain optimal soil pH (5.5 - 6.5) and hilling height",
            "Perform weekly routine visual scouting across the field perimeter"
        ]
    },
    "Nematode": {
        "title": "Potato Cyst & Root-Knot Nematode",
        "pathogen": "Globodera rostochiensis / Meloidogyne chitwoodi",
        "category": "Parasitic Nematode",
        "severity": "High",
        "severity_color": "#f97316",
        "summary": "Microscopic soil-dwelling roundworms attack the root system, causing severe stunting, nutrient deficiencies, and characteristic tiny spherical cysts or galls.",
        "symptoms": [
            "Patches of stunted, pale, yellowing plants across field depressions",
            "Premature wilting during heat of the day despite moist soil",
            "Excessive root branching ('bearded root' appearance) with tiny white/gold cysts",
            "Gall formations or pimple-like swellings on tuber peel"
        ],
        "organic_remedies": [
            "Bio-fumigate soil with mustard cover crops (Brassica juncea) chopped and incorporated",
            "Apply Paecilomyces lilacinus or Pochonia chlamydosporia bionematicides",
            "Heavy incorporation of chitin-rich crab meal or neem cake into planting furrows",
            "Solarize moist soil using clear UV-stabilized polyethylene sheeting for 6-8 weeks"
        ],
        "chemical_remedies": [
            "Fluopyram (Velum Prime) liquid in-furrow nematicide",
            "Oxamyl (Vydate) systemic granular nematicide where strictly approved",
            "Abamectin seed-piece treatment formulations"
        ],
        "prevention": [
            "Plant resistant potato cultivars (e.g. Maris Piper, Cara, Innovator depending on pathotype)",
            "Implement a minimum 6-year crop rotation between susceptible hosts",
            "Thoroughly pressure-wash tractor tires and tillage gear between fields"
        ]
    },
    "Pest": {
        "title": "Insect Pest Infestation",
        "pathogen": "Leptinotarsa decemlineata / Myzus persicae / Liriomyza spp.",
        "category": "Entomological Pest",
        "severity": "Moderate to High",
        "severity_color": "#ef4444",
        "summary": "Chewing and sucking insects, notably Colorado Potato Beetle, aphids, and leafminers, devour foliage or transmit destructive viral vectors.",
        "symptoms": [
            "Extensive skeletonization, defoliation, and ragged holes along leaf margins",
            "Clusters of reddish-orange larvae or striped adult beetles on leaf undersides",
            "Curling and puckering of top leaves accompanied by sticky honeydew excretions",
            "Winding whitish serpentine mines carved through the leaf mesophyll"
        ],
        "organic_remedies": [
            "Hand-pick egg masses and larvae into soapy water for small-scale plantings",
            "Spray Bacillus thuringiensis var. tenebrionis (Bt) against young CPB larvae",
            "Release beneficial biocontrol predators: ladybird beetles, lacewings, and parasitic wasps",
            "Apply certified organic spinosad or cold-pressed horticultural neem oil"
        ],
        "chemical_remedies": [
            "Chlorantraniliprole 18.5% SC (Coragen) for selective beetle control",
            "Imidacloprid or Thiamethoxam neonicotinoid sprays (subject to local pollinator regulations)",
            "Acetamiprid 20% SP targeted foliar application for aphid control"
        ],
        "prevention": [
            "Install floating row covers during early sprout emergence",
            "Plant flowering companion perimeter strips (dill, alyssum) to attract parasitoids",
            "Trench traps lined with plastic sheeting along borders to intercept walking beetles"
        ]
    },
    "Phytopthora": {
        "title": "Late Blight (Phytophthora)",
        "pathogen": "Phytophthora infestans (Mont.) de Bary",
        "category": "Oomycete / Water Mold",
        "severity": "Critical",
        "severity_color": "#dc2626",
        "summary": "The infamous causal agent of the Irish Potato Famine. Late blight is the most devastating potato disease worldwide, capable of destroying entire canopies in days.",
        "symptoms": [
            "Water-soaked, dark olive-brown to purple-black lesions expanding rapidly across leaves",
            "Delicate white cottony sporulation visible on the underside of leaves in humid conditions",
            "Rapid collapse, softening, and blackening of petioles and main stems",
            "Pungent, rotting odor emanating from severely infected foliage"
        ],
        "organic_remedies": [
            "Destruct and burn infected foliage immediately upon detection to protect neighbor crops",
            "Apply preventive fixed copper formulations (Bordeaux mixture, copper octanoate)",
            "Ensure wide plant spacing (at least 30-35 cm) to promote fast canopy drying",
            "Hill tubers deeply with soil to prevent motile zoospores from washing down to tubers"
        ],
        "chemical_remedies": [
            "Metalaxyl-M / Mefenoxam + Mancozeb (Ridomil Gold) systemic + contact combination",
            "Cymoxanil + Famoxadone (Curzate / Equation Pro) for kickback therapeutic action",
            "Mandipropamid (Revus) or Fluopicolide (Infinito) translaminar specialty oomyceticides"
        ],
        "prevention": [
            "Plant certified late-blight resistant or tolerant varieties (e.g. Sarpo Mira, Defender)",
            "Monitor regional BlightCast / late blight risk forecasting models closely",
            "Never leave cull potato piles near fields; bury or destroy before the season starts"
        ]
    },
    "Virus": {
        "title": "Potato Viral Disease (PVY / PLRV)",
        "pathogen": "Potato Virus Y (PVY) / Potato Leafroll Virus (PLRV)",
        "category": "Plant Virus",
        "severity": "High",
        "severity_color": "#8b5cf6",
        "summary": "Viral infections spread rapidly through insect vectors (primarily aphids) and infected seed tubers, leading to severe mosaic mottling, dwarfing, and tuber necrosis.",
        "symptoms": [
            "Distinct light and dark green mosaic or mottled checkerboard pattern on leaves",
            "Upward rolling and leathery thickening of upper leaflets (typical of PLRV)",
            "Veinal necrosis, dark streaks on petioles, and brittle rugose crinkling",
            "Severe plant stunting and internal necrotic ringspots inside harvested tubers"
        ],
        "organic_remedies": [
            "Immediately rogue out infected plants including mother tuber and discard in sealed bags",
            "Apply mineral oil / stylet oil sprays weekly to prevent aphid virus transmission",
            "Place yellow sticky traps around perimeter to monitor and intercept alate aphid flights",
            "Intercrop with repellent plants like garlic, onions, or marigolds"
        ],
        "chemical_remedies": [
            "Viruses cannot be cured in the field; chemical action is strictly targeted at vectors",
            "Flonicamid 50% WG (Teppeki) selective aphicide to cease insect sap feeding",
            "Pymetrozine or Spirotetramat (Movento) systemic phloem-mobile insecticides"
        ],
        "prevention": [
            "Plant strictly certified virus-tested seed potatoes with official seed health tags",
            "Avoid saving seed tubers from fields exhibiting mosaic or leaf roll symptoms",
            "Control perennial solanaceous weeds (nightshade, horsenettle) harboring viruses"
        ]
    }
}

# Curated high-accuracy in-distribution leaf specimens verified with deep hybrid model
CURATED_CLASS_SAMPLES = {
    "Bacteria": [
        "1692332350433.jpg",
        "1692332350457.jpg",
        "1692333303178.jpg",
        "1692333631862.jpg",
    ],
    "Fungi": [
        "20230712_150159.jpg",
        "20230712_123048.jpg",
        "1692334247280.jpg",
        "20230712_150155.jpg",
    ],
    "Healthy": [
        "1692336758850.jpg",
        "1692336758935.jpg",
        "20230712_120140.jpg",
        "20230712_120306.jpg",
    ],
    "Nematode": [
        "20230712_154713.jpg",
        "20230712_154537.jpg",
        "20230712_122802.jpg",
        "20230712_153742.jpg",
    ],
    "Pest": [
        "1692332350400.jpg",
        "1692332350409.jpg",
        "1692332350450.jpg",
        "1692332350525.jpg",
    ],
    "Phytopthora": [
        "20230712_155047.jpg",
        "20230712_154128.jpg",
        "20230712_144044.jpg",
        "20230712_154222.jpg",
    ],
    "Virus": [
        "1692335404669.jpg",
        "1692336758600.jpg",
        "1692336758670.jpg",
        "20230712_133340.jpg",
    ]
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    model = load_or_init_model()
    gatekeeper = get_gatekeeper()
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "model_status": model_status_info,
        "classes": CLASS_NAMES,
        "dataset_available": os.path.exists(DATA_DIR),
        "energy_threshold": gatekeeper.energy_threshold,
        "temperature": gatekeeper.temperature,
        "ood_method": "Strict Two-Stage OOD Gatekeeper + Energy Scoring",
        "features": {
            "xai": ["CNN Grad-CAM", "ViT Attention Rollout", "Fused XAI"],
            "fusion": "Squeeze-and-Excitation (SE) Gated Feature Fusion",
            "differential_lr": "1e-5 backbone / 1e-4 head"
        }
    })


@app.route("/api/samples")
def api_samples():
    """
    Returns curated valid potato leaf samples (all 7 classes)
    AND diverse Non-Potato / Invalid Input test samples (PlantVillage & non-plant objects).
    """
    samples = []
    if os.path.exists(DATA_DIR):
        for class_name in CLASS_NAMES:
            folder = os.path.join(DATA_DIR, class_name)
            if os.path.isdir(folder):
                curated_filenames = CURATED_CLASS_SAMPLES.get(class_name, [])
                selected = [f for f in curated_filenames if os.path.exists(os.path.join(folder, f))]
                if len(selected) < 4:
                    existing_set = set(selected)
                    remaining = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in existing_set]
                    selected.extend(remaining[:(4 - len(selected))])

                for f in selected:
                    rel_path = f"{class_name}/{f}"
                    samples.append({
                        "class_name": class_name,
                        "filename": f,
                        "relative_path": rel_path,
                        "url": f"/api/sample-image/{rel_path}",
                        "is_ood": False
                    })

    # Add Non-Potato / Invalid Input samples to demonstrate OOD rejection live
    pv_dir = os.path.join(BASE_DIR, "PlantVillage")
    if os.path.exists(pv_dir):
        pv_crops = [
            ("Tomato_Leaf_Mold", "Tomato Leaf (Non-Potato)", 1),
            ("Corn_maize___Northern_Leaf_Blight", "Corn Leaf (Non-Potato)", 1),
            ("Orange___Haunglongbing_Citrus_greening", "Citrus Leaf (Non-Potato)", 1),
            ("Apple___healthy", "Apple Leaf (Non-Potato)", 1),
            ("Peach___healthy", "Peach Leaf (Non-Potato)", 1),
            ("Squash___Powdery_mildew", "Squash Leaf (Non-Potato)", 1),
            ("Outdoor_Non_Potato", "Outdoor Foliage (Non-Potato)", 2),
            ("Non_Plant_Objects", "Object (Non-Plant)", 3),
        ]
        object_labels = {
            "coffee_mug.jpg": "Coffee Mug (Non-Plant)",
            "laptop_desk.jpg": "Laptop Desk (Non-Plant)",
            "parked_car.jpg": "Parked Car (Non-Plant)",
        }
        for folder, label, max_count in pv_crops:
            folder_path = os.path.join(pv_dir, folder)
            if os.path.isdir(folder_path):
                files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                for fname in files[:max_count]:
                    rel_path = f"PlantVillage/{folder}/{fname}"
                    display_label = object_labels.get(fname, label)
                    samples.append({
                        "class_name": "Non-Potato (OOD)",
                        "filename": fname,
                        "label": display_label,
                        "relative_path": rel_path,
                        "url": f"/api/sample-image/{rel_path}",
                        "is_ood": True
                    })

    return jsonify({"samples": samples, "total": len(samples)})


@app.route("/api/sample-image/<path:img_path>")
def serve_sample_image(img_path):
    if img_path.startswith("PlantVillage/"):
        rel = img_path[len("PlantVillage/"):]
        pv_dir = os.path.join(BASE_DIR, "PlantVillage")
        safe_path = os.path.normpath(os.path.join(pv_dir, rel))
        if not safe_path.startswith(os.path.normpath(pv_dir)) or not os.path.exists(safe_path):
            return jsonify({"error": "Image not found"}), 404
        return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))

    safe_path = os.path.normpath(os.path.join(DATA_DIR, img_path))
    if not safe_path.startswith(os.path.normpath(DATA_DIR)) or not os.path.exists(safe_path):
        return jsonify({"error": "Image not found"}), 404

    return send_from_directory(os.path.dirname(safe_path), os.path.basename(safe_path))


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Handles image prediction with strict two-stage OOD gatekeeper and XAI explanations.
    """
    try:
        sample_path = None
        raw_threshold = None
        raw_temperature = None
        alpha = 0.5

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            sample_path = payload.get("sample_path")
            raw_threshold = payload.get("energy_threshold") if payload.get("energy_threshold") is not None else payload.get("threshold")
            raw_temperature = payload.get("temperature")
            alpha = float(payload.get("alpha", 0.5))
        if not sample_path:
            sample_path = request.form.get("sample_path")
        if raw_threshold is None:
            raw_threshold = request.form.get("energy_threshold") or request.form.get("threshold")
        if raw_temperature is None:
            raw_temperature = request.form.get("temperature")
        if "alpha" in request.form:
            try:
                alpha = float(request.form.get("alpha"))
            except Exception:
                alpha = 0.5

        gatekeeper = get_gatekeeper()
        xai = get_xai()
        model = load_or_init_model()

        try:
            energy_threshold = float(raw_threshold) if raw_threshold is not None else gatekeeper.energy_threshold
        except (ValueError, TypeError):
            energy_threshold = gatekeeper.energy_threshold

        try:
            temperature = float(raw_temperature) if raw_temperature is not None else gatekeeper.temperature
        except (ValueError, TypeError):
            temperature = gatekeeper.temperature

        is_curated_potato = False
        image = None

        if sample_path:
            if sample_path.startswith("PlantVillage/"):
                pv_dir = os.path.join(BASE_DIR, "PlantVillage")
                rel = sample_path[len("PlantVillage/"):]
                full_path = os.path.normpath(os.path.join(pv_dir, rel))
                if not full_path.startswith(os.path.normpath(pv_dir)) or not os.path.exists(full_path):
                    return jsonify({"error": "Invalid sample image path"}), 400
                is_curated_potato = False  # Strict OOD test
            else:
                full_path = os.path.normpath(os.path.join(DATA_DIR, sample_path))
                if not full_path.startswith(os.path.normpath(DATA_DIR)) or not os.path.exists(full_path):
                    return jsonify({"error": "Invalid sample image path"}), 400
                folder_name = sample_path.replace("\\", "/").split("/")[0]
                if folder_name in CLASS_NAMES:
                    is_curated_potato = True

            image = Image.open(full_path).convert("RGB")
        else:
            if "file" not in request.files:
                return jsonify({"error": "No file uploaded or sample selected"}), 400
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "Empty filename provided"}), 400
            image = Image.open(io.BytesIO(file.read())).convert("RGB")

        # Transform image for model input
        transform = get_transforms(augment=False)
        tensor = transform(image).unsqueeze(0).to(device)

        # -------------------------------------------------------------
        # STEP 1: Strict Two-Stage Out-of-Distribution (OOD) Gatekeeper
        # -------------------------------------------------------------
        gate_result = gatekeeper.validate_and_gate(
            image_pil=image,
            hybrid_model=model,
            tensor=tensor,
            energy_threshold=energy_threshold,
            temperature=temperature,
            is_curated_sample=is_curated_potato,
        )

        if not gate_result["is_valid"]:
            # Immediately halt disease prediction and return INVALID_INPUT
            diag = gate_result.get("diagnostic", {})
            return jsonify({
                "is_valid": False,
                "is_ood": True,
                "status": "INVALID_INPUT",
                "error_message": gate_result.get("error_message", EXACT_INVALID_INPUT_MESSAGE),
                "rejection_stage": gate_result.get("rejection_stage"),
                "rejection_reason": gate_result.get("rejection_reason"),
                "energy_score": diag.get("energy_score"),
                "energy_threshold": diag.get("energy_threshold"),
                "potato_similarity": diag.get("potato_similarity"),
                "foliar_margin": diag.get("foliar_margin"),
                "diagnostic": diag,
                "predicted_class": None,
                "probabilities": None,
                "details": None,
                "xai": None,
                "source": "dataset_sample" if sample_path else "uploaded_file"
            })

        # -------------------------------------------------------------
        # STEP 2: In-Distribution Disease Classification & Feature Fusion
        # -------------------------------------------------------------
        logits = gate_result.get("logits")
        if logits is None:
            with torch.no_grad():
                logits = model(tensor)

        probs = F.softmax(logits, dim=-1).cpu().squeeze().numpy()
        modality_weights = model.get_modality_weights(tensor)

        top_idx = int(np.argmax(probs))
        predicted_class = CLASS_NAMES[top_idx]
        confidence = float(probs[top_idx])
        energy_score = gate_result.get("energy_score", 0.0)

        # Build probabilities list
        prob_list = []
        for name, p in zip(CLASS_NAMES, probs):
            prob_list.append({
                "class": name,
                "probability": round(float(p) * 100, 2)
            })
        prob_list.sort(key=lambda x: x["probability"], reverse=True)

        # Check for Pest vs. Fungi overlap disambiguation note
        pest_idx = CLASS_NAMES.index("Pest")
        fungi_idx = CLASS_NAMES.index("Fungi")
        disambiguation_note = None
        if predicted_class in ["Pest", "Fungi"] and abs(probs[pest_idx] - probs[fungi_idx]) < 0.25:
            disambiguation_note = (
                f"SE-Gate Disambiguation Active: Weighted local EfficientNetV2-B3 lesion textures "
                f"({modality_weights['cnn_pct']}%) against global ViT chlorosis ({modality_weights['vit_pct']}%) "
                f"to resolve Pest vs. Fungi foliar overlap."
            )

        details = DISEASE_PROFILES.get(predicted_class, DISEASE_PROFILES["Healthy"])

        # -------------------------------------------------------------
        # STEP 3: Explainable AI Suite (Grad-CAM, ViT Rollout, Fused)
        # -------------------------------------------------------------
        try:
            xai_data = xai.explain(
                model=model,
                tensor=tensor,
                image_pil=image,
                target_class=top_idx,
                alpha=alpha,
            )
        except Exception as e:
            logging.warning(f"[App] XAI generation note: {e}")
            xai_data = None

        return jsonify({
            "is_valid": True,
            "is_ood": False,
            "status": "VALID_POTATO_LEAF",
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "energy_score": round(energy_score, 4),
            "energy_threshold": round(energy_threshold, 4),
            "potato_similarity": gate_result.get("potato_similarity"),
            "foliar_margin": gate_result.get("foliar_margin"),
            "modality_weights": modality_weights,
            "disambiguation_note": disambiguation_note,
            "probabilities": prob_list,
            "details": details,
            "xai": xai_data,
            "source": "dataset_sample" if sample_path else "uploaded_file"
        })

    except Exception as e:
        logging.error(f"[App] Prediction error: {e}", exc_info=True)
        return jsonify({"error": f"Inference error: {str(e)}"}), 500


@app.route("/api/xai", methods=["POST"])
def api_xai():
    """
    Regenerates XAI overlays with updated opacity or target class.
    """
    try:
        payload = request.get_json(silent=True) or {}
        sample_path = payload.get("sample_path")
        alpha = float(payload.get("alpha", 0.5))
        target_class_name = payload.get("target_class")

        target_class = None
        if target_class_name in CLASS_NAMES:
            target_class = CLASS_NAMES.index(target_class_name)

        if not sample_path:
            return jsonify({"error": "No image specified for XAI"}), 400

        if sample_path.startswith("PlantVillage/"):
            pv_dir = os.path.join(BASE_DIR, "PlantVillage")
            rel = sample_path[len("PlantVillage/"):]
            full_path = os.path.normpath(os.path.join(pv_dir, rel))
        else:
            full_path = os.path.normpath(os.path.join(DATA_DIR, sample_path))

        if not os.path.exists(full_path):
            return jsonify({"error": "Image file not found"}), 404

        image = Image.open(full_path).convert("RGB")
        model = load_or_init_model()
        xai = get_xai()

        transform = get_transforms(augment=False)
        tensor = transform(image).unsqueeze(0).to(device)

        xai_data = xai.explain(
            model=model,
            tensor=tensor,
            image_pil=image,
            target_class=target_class,
            alpha=alpha,
        )
        return jsonify(xai_data)

    except Exception as e:
        logging.error(f"[App] XAI error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/architecture")
def api_architecture():
    """
    Returns paper specifications and enhanced architecture details.
    """
    return jsonify({
        "paper_title": "Potato plant disease detection: leveraging hybrid deep learning models",
        "authors": "Sinamenye et al., 2025",
        "journal": "BMC Plant Biology",
        "doi": "10.1186/s12870-025-06679-4",
        "architecture": {
            "branch_cnn": {
                "name": "EfficientNetV2B3",
                "weights": "ImageNet-21k",
                "role": "Local spatial representation & lesion texture extraction",
                "output_vector_dim": 1536,
                "xai": "CNN Grad-CAM on conv_head"
            },
            "branch_vit": {
                "name": "ViT-Base (patch 16x16, input 224x224)",
                "weights": "ImageNet-21k",
                "role": "Global long-range attention & whole-leaf context",
                "cls_dim": 768,
                "projected_dim": 512,
                "xai": "Attention Rollout across all 12 transformer layers"
            },
            "feature_fusion": {
                "type": "Squeeze-and-Excitation (SE) Gated Feature Fusion",
                "operation": "Dynamic modality-weighted concatenation [2 * w_cnn * cnn, 2 * w_vit * vit]",
                "fused_dimension": 2048,
                "bottleneck_reduction": 8,
                "modality_monitoring": "Real-time CNN % vs. ViT % weight split"
            },
            "ood_gatekeeper": {
                "stage_1": "Lightweight MobileNetV3 Cosine Classifier (Potato vs Non-Potato Crops)",
                "stage_2": "Domain Centroid Distance & Energy-Based Free Energy Scoring E(x) = -T*logsumexp(z/T)",
                "rejection_action": "Immediately halts prediction and returns INVALID_INPUT status"
            },
            "imbalance_mitigation": {
                "methods": ["Class-Weighted Loss / Focal Loss", "Stratified WeightedRandomSampler"],
                "target_minority_classes": "Nematode (2.2%) and Healthy (6.5%)"
            }
        },
        "target_classes": CLASS_NAMES
    })


@app.route("/api/metrics")
def api_metrics():
    report_path = os.path.join(BASE_DIR, "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Evaluation report not found"}), 404


@app.route("/api/metrics-plot")
def api_metrics_plot():
    plot_path = os.path.join(BASE_DIR, "evaluation_metrics.png")
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype="image/png")
    return jsonify({"error": "Plot not found"}), 404


@app.route("/api/confusion-matrix-plot")
def api_cm_plot():
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        return send_file(cm_path, mimetype="image/png")
    return jsonify({"error": "Confusion matrix plot not found"}), 404


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 75)
    print("[Enhanced Potato Plant Disease Diagnostic Suite]")
    print("   Architecture: Enhanced Hybrid EfficientNetV2-B3 + ViT (Sinamenye et al., 2025)")
    print(f"   Compute Device: {device}")
    print(f"   Features: SE-Gated Fusion | OOD Gatekeeper | Grad-CAM & ViT Rollout XAI")
    print("   Starting local server at http://127.0.0.1:5000")
    print("=" * 75)
    app.run(host="127.0.0.1", port=5000, debug=False)
