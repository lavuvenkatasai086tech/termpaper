"""
Root compatibility wrapper for train.py.
Imports training pipeline from src.train.
"""
from src.train import (
    train_model,
    evaluate,
    compute_metrics,
    FocalLoss,
)

__all__ = [
    "train_model",
    "evaluate",
    "compute_metrics",
    "FocalLoss",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Enhanced Hybrid EfficientNet-ViT")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--loss_type", type=str, default="focal", choices=["cross_entropy", "weighted_ce", "focal"])
    parser.add_argument("--unfreeze_top_blocks", action="store_true", default=False)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss_type,
        unfreeze_top_blocks=args.unfreeze_top_blocks,
    )
