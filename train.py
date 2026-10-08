import os
import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from dataset import create_dataloaders
from model import EfficientNetV2B3ViT


def compute_metrics(y_true, y_pred):
    """
    Computes Accuracy, Macro Precision, Macro Recall, Macro F1, and MCC.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png", title="Confusion Matrix"):
    """
    Generates and saves a publication-quality confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Confusion matrix saved to: {save_path}")


def evaluate(model, data_loader, criterion, device, class_names=None, cm_save_path=None):
    """
    Evaluates the model over a dataloader.
    Returns average loss, metric dict, y_true, and y_pred.
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

    if cm_save_path and class_names is not None:
        plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_save_path)

    return metrics, y_true, y_pred


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    class_names,
    device,
    epochs: int = 70,
    lr: float = 0.0001,
    patience: int = 5,
    factor: float = 0.5,
    save_dir: str = "checkpoints",
    use_wandb: bool = False,
):
    """
    Executes the training and validation loops following the paper specifications:
    - Loss: CrossEntropyLoss
    - Optimizer: Adam (lr = 1e-4)
    - Scheduler: ReduceLROnPlateau (factor = 0.5, patience = 5 on validation loss)
    """
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # Only train parameters that require gradients (the classifier head and projection layer)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        verbose=True,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_checkpoint_path = os.path.join(save_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(save_dir, "last_model.pth")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1": [], "val_mcc": [],
        "lr": []
    }

    print("\n" + "=" * 80)
    print(f"Starting Training: {epochs} Epochs | Batch Size: {train_loader.batch_size} | Device: {device}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
    print("=" * 80 + "\n")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch_idx, (images, labels) in enumerate(train_loader):
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

        # Validation phase
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        val_mcc = val_metrics["mcc"]

        # Step LR scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_mcc"].append(val_mcc)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} - F1: {val_f1:.4f} - MCC: {val_mcc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_mcc": val_mcc,
                "lr": current_lr,
            })

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_metrics": val_metrics,
                "class_names": class_names,
            }, best_checkpoint_path)
            print(f"  --> Best model checkpoint saved to: {best_checkpoint_path} (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")

    # Save last model checkpoint
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
    }, last_checkpoint_path)
    print(f"\n[Training Complete] Last model checkpoint saved to: {last_checkpoint_path}")
    print(f"Total training time: {(time.time() - start_time) / 60:.2f} minutes")

    # Save training history
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # Final evaluation on Test Set using Best Model Checkpoint
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION (Using Best Checkpoint)")
    print("=" * 80)
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    cm_path = os.path.join(save_dir, "test_confusion_matrix.png")
    test_metrics, y_true, y_pred = evaluate(
        model, test_loader, criterion, device, class_names=class_names, cm_save_path=cm_path
    )

    print("\nTest Set Metrics:")
    print(f"  - Test Loss:      {test_metrics['loss']:.4f}")
    print(f"  - Accuracy:       {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Macro Precision:{test_metrics['precision']:.4f}")
    print(f"  - Macro Recall:   {test_metrics['recall']:.4f}")
    print(f"  - Macro F1-Score: {test_metrics['f1']:.4f}")
    print(f"  - MCC:            {test_metrics['mcc']:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    return test_metrics


def main():
    parser = argparse.ArgumentParser(
        description="EfficientNetV2B3 + ViT Hybrid Training for Potato Disease Detection"
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to potato dataset folder")
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs (default: 70)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate (default: 0.0001)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0 for Windows)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--eval_only", type=str, default=None, help="Path to checkpoint .pth to evaluate without training")
    parser.add_argument("--quick_test", action="store_true", help="Run 1 epoch with small batch for sanity checking")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    batch_size = 8 if args.quick_test else args.batch_size
    epochs = 1 if args.quick_test else args.epochs

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")

    model = EfficientNetV2B3ViT(num_classes=num_classes).to(device)

    if args.eval_only:
        print(f"Evaluating checkpoint: {args.eval_only}")
        checkpoint = torch.load(args.eval_only, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        criterion = nn.CrossEntropyLoss()
        cm_path = os.path.join(args.save_dir, "eval_confusion_matrix.png")
        metrics, y_true, y_pred = evaluate(model, test_loader, criterion, device, class_names, cm_save_path=cm_path)
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
        return

    if args.wandb:
        try:
            import wandb
            wandb.init(project="potato-disease-hybrid", config=vars(args))
        except Exception as e:
            print(f"[Warning] Failed to initialize wandb: {e}. Continuing with console logging.")
            args.wandb = False

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        epochs=epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
