import os
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

from .dataset import create_dataloaders
from .model import EnhancedHybridEfficientNetViT


class FocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Counteracts severe class imbalance (Nematode 2.2%, Healthy 6.5%) by
    dynamically down-weighting easy examples and focusing gradients on hard/minority instances.
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes Accuracy, Macro Precision, Macro Recall, Macro F1, and MCC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "mcc": float(mcc),
    }


def evaluate(model, data_loader, criterion, device):
    """
    Evaluates model over DataLoader, returning average loss, metrics dictionary, y_true, y_pred.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    total_samples = len(all_targets)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = avg_loss
    return metrics, y_true, y_pred


def train_model(
    config_path: str = None,
    data_dir: str = None,
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    backbone_lr: float = None,
    loss_type: str = None,
    focal_gamma: float = 2.0,
    use_weighted_sampler: bool = None,
    unfreeze_top_blocks: bool = None,
    save_dir: str = "checkpoints",
):
    """
    Executes training and validation pipeline incorporating:
    - Squeeze-and-Excitation Gated Feature Fusion
    - Differential learning rates for partial backbone unfreezing (1e-5 vs 1e-4)
    - Focal Loss / Weighted Cross-Entropy & Stratified WeightedRandomSampler
    """
    # Load config defaults
    base_dir = os.path.dirname(os.path.dirname(__file__))
    if config_path is None:
        config_path = os.path.join(base_dir, "configs", "config.yaml")

    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    epochs = epochs or train_cfg.get("epochs", 70)
    batch_size = batch_size or train_cfg.get("batch_size", 64)
    lr = lr or train_cfg.get("learning_rate", 1e-4)
    backbone_lr = backbone_lr or train_cfg.get("backbone_learning_rate", 1e-5)
    loss_type = loss_type or train_cfg.get("loss_type", "focal")
    use_weighted_sampler = use_weighted_sampler if use_weighted_sampler is not None else train_cfg.get("use_weighted_sampler", True)
    unfreeze_top_blocks = unfreeze_top_blocks if unfreeze_top_blocks is not None else model_cfg.get("unfreeze_top_blocks", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"Training EnhancedHybridEfficientNetViT | Device: {device}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Head LR: {lr} | Backbone LR: {backbone_lr}")
    print(f"Loss: {loss_type} | Weighted Sampler: {use_weighted_sampler} | Unfreeze Top Blocks: {unfreeze_top_blocks}")
    print("=" * 80 + "\n")

    # Prepare DataLoaders
    train_loader, val_loader, test_loader, class_names, class_weights = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
    )
    class_weights = class_weights.to(device)

    # Instantiate Model
    model = EnhancedHybridEfficientNetViT(
        num_classes=len(class_names),
        unfreeze_top_blocks=unfreeze_top_blocks,
        use_se_gating=True,
    ).to(device)

    # Loss selection
    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    elif loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer with differential learning rates
    param_groups = model.get_parameter_groups(backbone_lr=backbone_lr, head_lr=lr)
    optimizer = optim.Adam(param_groups)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
    )

    os.makedirs(save_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(save_dir, "best_model.pth")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        train_preds, train_targets = [], []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_targets)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        val_mcc = val_metrics["mcc"]

        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} MCC: {val_mcc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_metrics": val_metrics,
                "class_names": class_names,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path}")

    print("\nTraining complete. Running final evaluation on test set...")
    test_metrics, _, _ = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Test F1: {test_metrics['f1']*100:.2f}% | Test MCC: {test_metrics['mcc']:.4f}")
    return model, test_metrics


if __name__ == "__main__":
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
