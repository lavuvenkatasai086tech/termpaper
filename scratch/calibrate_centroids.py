import os
import torch
import numpy as np

# Load cache and model checkpoint
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

cache = torch.load('checkpoints/features_cache_full.pt', map_location='cpu')
eff_feats = cache['eff_feats']  # [3076, 1536]
vit_cls = cache['vit_cls']      # [3076, 768]
labels = cache['labels']        # [3076]

vit_w = sd['vit_linear.weight']
vit_b = sd['vit_linear.bias']
vit_proj = torch.matmul(vit_cls, vit_w.t()) + vit_b
fused = torch.cat([eff_feats, vit_proj], dim=1) # [3076, 2048]
fused_norm = fused / torch.norm(fused, dim=1, keepdim=True)

# 1. Global centroid
global_centroid = fused_norm.mean(dim=0, keepdim=True)
global_centroid = global_centroid / torch.norm(global_centroid, dim=1, keepdim=True)

# 2. Per-class centroids
class_centroids = []
for c in range(7):
    mask = (labels == c)
    c_feats = fused_norm[mask]
    c_cent = c_feats.mean(dim=0, keepdim=True)
    c_cent = c_cent / torch.norm(c_cent, dim=1, keepdim=True)
    class_centroids.append(c_cent)
class_centroids = torch.cat(class_centroids, dim=0) # [7, 2048]

# Calculate cosine similarities for all 3076 in-distribution samples
global_sims = torch.matmul(fused_norm, global_centroid.t()).squeeze().numpy()
max_class_sims = torch.matmul(fused_norm, class_centroids.t()).max(dim=1).values.numpy()

print(f"Total In-Distribution Samples: {len(labels)}")
print(f"Global Centroid CosSim:")
print(f"  Min: {global_sims.min():.4f}, 1%: {np.percentile(global_sims, 1):.4f}, 5%: {np.percentile(global_sims, 5):.4f}, Mean: {global_sims.mean():.4f}")
print(f"Max Class Centroid CosSim:")
print(f"  Min: {max_class_sims.min():.4f}, 1%: {np.percentile(max_class_sims, 1):.4f}, 5%: {np.percentile(max_class_sims, 5):.4f}, Mean: {max_class_sims.mean():.4f}")

# Also check energy distribution of in-distribution samples
classifier_w = sd['classifier.weight']
classifier_b = sd['classifier.bias']
logits = torch.matmul(fused, classifier_w.t()) + classifier_b
energies = - torch.logsumexp(logits, dim=-1).numpy()
print(f"Energy Distribution (In-Distribution):")
print(f"  Min: {energies.min():.4f}, Mean: {energies.mean():.4f}, 95%: {np.percentile(energies, 95):.4f}, 99%: {np.percentile(energies, 99):.4f}, Max: {energies.max():.4f}")

# Save centroids file for ultra-fast OOD gating
ood_data = {
    'global_centroid': global_centroid.squeeze(0), # [2048]
    'class_centroids': class_centroids,            # [7, 2048]
    'global_sim_min': float(global_sims.min()),
    'global_sim_p01': float(np.percentile(global_sims, 1)),
    'max_class_sim_min': float(max_class_sims.min()),
    'max_class_sim_p01': float(np.percentile(max_class_sims, 1)),
    'energy_p99': float(np.percentile(energies, 99)),
}
torch.save(ood_data, 'checkpoints/potato_ood_centroids.pt')
print("Successfully generated and saved checkpoints/potato_ood_centroids.pt!")
