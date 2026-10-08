# 🚀 End-to-End GPU Training Guide (Google Colab / Kaggle)

To train the **EfficientNetV2B3 + Vision Transformer (ViT)** model end-to-end and reproduce the **85.06%** accuracy benchmark from the paper (*Sinamenye et al., 2025*):

---

## ⚡ Option 1: Run on Google Colab (Free T4 GPU, ~30–45 mins)

### Step 1: Open Google Colab & Select GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. In the top menu, go to **Runtime** &rarr; **Change runtime type** &rarr; Select **T4 GPU** &rarr; Click **Save**.

### Step 2: Upload Project Files
In the left sidebar file browser:
1. Upload `Potato Leaf Disease Dataset in Uncontrolled Environment.zip` (or your dataset folder).
2. Upload the project code files:
   - `model.py`
   - `dataset.py`
   - `train_end_to_end_gpu.py`
   - `weights/` folder (or let PyTorch download them automatically via timm/transformers).

### Step 3: Run Training Command
Run the following cells in Colab:

```bash
# 1. Install required packages
!pip install -q timm transformers scikit-learn seaborn matplotlib

# 2. Launch End-to-End GPU Training (70 Epochs)
!python train_end_to_end_gpu.py --epochs 70 --warmup_epochs 10 --batch_size 32
```

### Step 4: Download Trained Outputs
Once training finishes:
- `checkpoints/best_model.pth` (~400MB)
- `evaluation_report.json`
- `evaluation_metrics.png`
- `confusion_matrix.png`

Download `best_model.pth` and place it back into your local `checkpoints/` folder. Your local web app at `http://127.0.0.1:5000` will immediately load the new weights!

---

## ⚡ Option 2: Run Locally (if NVIDIA GPU with CUDA is available)

Open your terminal or PowerShell with CUDA enabled:

```powershell
python train_end_to_end_gpu.py --epochs 70 --warmup_epochs 10 --batch_size 32
```

---

## 📊 Expected Performance Milestones
- **Epochs 1–10 (Stage 1 - Linear Warmup)**: Rapid convergence of hybrid projection head to ~76–78% accuracy.
- **Epochs 11–70 (Stage 2 - Backbone Fine-Tuning)**: Unfreezing top convolution and transformer self-attention blocks with differential learning rates (`lr_backbone=1.5e-5`, `lr_head=2e-4`) pushes accuracy past **85%**.
