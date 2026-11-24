"""
Root compatibility wrapper for dataset.py.
Imports dataset and transform utilities from src.dataset.
"""
from src.dataset import (
    get_transforms,
    PotatoLeafDataset,
    create_dataloaders,
    compute_class_weights,
    create_stratified_weighted_sampler,
    auto_prepare_data_dir,
)

__all__ = [
    "get_transforms",
    "PotatoLeafDataset",
    "create_dataloaders",
    "compute_class_weights",
    "create_stratified_weighted_sampler",
    "auto_prepare_data_dir",
]

if __name__ == "__main__":
    print("[Testing dataset.py compatibility wrapper]")
    train_loader, val_loader, test_loader, classes, weights = create_dataloaders(batch_size=16)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Class Weights: {weights.numpy()}")
