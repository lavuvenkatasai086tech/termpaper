import os
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from PIL import Image


def get_transforms(augment: bool = True, image_size: int = 224, resize_size: int = 256):
    """
    Constructs the data preprocessing and augmentation pipeline as specified in:
    'Potato plant disease detection: leveraging hybrid deep learning models' (Sinamenye et al., 2025).

    Pipeline Specs:
    - Initial Resize: 256x256
    - Augmentations:
      * Random crop to 224x224 (scale: 0.95-1.05, aspect ratio: 0.75-1.33, BICUBIC interpolation)
      * Random Horizontal & Vertical flips
      * Random rotation between -40° and +40°
      * Saturation adjustment (factor: 0.8) and Hue adjustment (factor: 0.021)
      * Random horizontal/vertical affine translation (factor: 0.13) and scale zoom (0.8-1.2)
    - Normalization: ImageNet mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.95, 1.05),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-40, 40)),
            transforms.ColorJitter(saturation=0.8, hue=0.021),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.13, 0.13),
                scale=(0.8, 1.2),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


class PotatoLeafDataset(Dataset):
    """
    Custom PyTorch Dataset that loads image paths with associated labels
    and applies split-specific transforms independently.
    """
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def compute_class_weights(targets: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """
    Computes inverse frequency class weights to counteract severe class imbalance
    reported in Sinamenye et al. (Nematode: 2.2%, Healthy: 6.5%).
    Weights: w_c = N / (C * N_c)
    """
    class_counts = np.bincount(targets, minlength=num_classes)
    total_samples = len(targets)
    weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(float))
    # Normalize weights so mean is 1.0
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def create_stratified_weighted_sampler(targets: np.ndarray, num_classes: int = 7) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that samples underrepresented classes
    (e.g., Nematode and Healthy) more frequently during training epochs.
    """
    class_counts = np.bincount(targets, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(float)
    sample_weights = class_weights[targets]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def auto_prepare_data_dir(data_dir: str = None) -> str:
    """
    Locates the dataset directory or extracts it from the local zip archive if needed.
    """
    candidate_paths = [
        data_dir,
        os.path.join("data", "potatodata", "Potato Leaf Disease Dataset in Uncontrolled Environment"),
        os.path.join("data", "potatodata"),
        os.path.join("..", "data", "potatodata"),
        os.path.join("/content", "data", "potatodata"),
        os.path.join("/content", "termpaper", "data", "potatodata"),
        "Potato Leaf Disease Dataset in Uncontrolled Environment",
        os.path.join("/content", "Potato Leaf Disease Dataset in Uncontrolled Environment"),
    ]
    for path in candidate_paths:
        if path and os.path.isdir(path) and len(os.listdir(path)) > 0:
            subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
            if len(subdirs) >= 3:
                return path
            if len(subdirs) == 1:
                nested = os.path.join(path, subdirs[0])
                nested_subdirs = [d for d in os.listdir(nested) if os.path.isdir(os.path.join(nested, d))]
                if len(nested_subdirs) >= 3:
                    return nested

    zip_candidates = [
        "potatodata_compact.zip",
        os.path.join("..", "potatodata_compact.zip"),
        os.path.join("/content", "potatodata_compact.zip"),
        "compact.zip",
        os.path.join("..", "compact.zip"),
        os.path.join("/content", "compact.zip"),
        "Potato Leaf Disease Dataset in Uncontrolled Environment.zip",
        os.path.join("..", "Potato Leaf Disease Dataset in Uncontrolled Environment.zip"),
        os.path.join("/content", "Potato Leaf Disease Dataset in Uncontrolled Environment.zip"),
        "potatodata.zip",
        os.path.join("..", "potatodata.zip"),
        os.path.join("/content", "potatodata.zip"),
    ]
    search_dirs = [".", "..", "/content"]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for fname in os.listdir(sdir):
                if fname.lower().endswith(".zip") and any(k in fname.lower() for k in ["compact", "potato", "leaf", "plant"]):
                    full_p = os.path.join(sdir, fname)
                    if full_p not in zip_candidates:
                        zip_candidates.append(full_p)

    for zip_path in zip_candidates:
        if os.path.isfile(zip_path):
            target_extract_dir = os.path.join("data", "potatodata")
            os.makedirs(target_extract_dir, exist_ok=True)
            print(f"[Dataset] Found archive '{zip_path}'. Extracting into '{target_extract_dir}'...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    parts = member.replace('\\', '/').split('/')
                    if len(parts) >= 2 and parts[-1]:
                        class_folder = parts[-2]
                        filename = parts[-1]
                        out_class_dir = os.path.join(target_extract_dir, class_folder)
                        os.makedirs(out_class_dir, exist_ok=True)
                        out_file_path = os.path.join(out_class_dir, filename)
                        with zip_ref.open(member) as source, open(out_file_path, "wb") as target:
                            target.write(source.read())
            print(f"[Dataset] Extracted dataset to '{target_extract_dir}'.")
            return target_extract_dir

    raise FileNotFoundError(
        "Could not find dataset directory or 'Potato Leaf Disease Dataset in Uncontrolled Environment.zip'."
    )


def create_dataloaders(
    data_dir: str = None,
    batch_size: int = 64,
    test_size: float = 0.1,
    val_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 0,
    augment: bool = True,
    use_weighted_sampler: bool = True,
):
    """
    Prepares train, validation, and test DataLoaders with stratified splits,
    properly separated transforms, and optional Stratified WeightedRandomSampler.

    Returns:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    valid_data_dir = auto_prepare_data_dir(data_dir)

    base_dataset = datasets.ImageFolder(valid_data_dir)
    class_names = base_dataset.classes
    samples = base_dataset.samples  # list of (path, class_idx)
    targets = np.array([s[1] for s in samples])

    # Stratified train/test split (90% train+val, 10% test)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(samples)),
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )

    # Stratified train/val split (80% train, 10% val overall)
    train_targets = targets[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1.0 - test_size),
        random_state=random_state,
        stratify=train_targets,
    )

    # Split sample lists
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_split_targets = np.array([s[1] for s in train_samples])
    class_weights = compute_class_weights(train_split_targets, num_classes=len(class_names))

    train_transform = get_transforms(augment=augment)
    eval_transform = get_transforms(augment=False)

    train_dataset = PotatoLeafDataset(train_samples, transform=train_transform)
    val_dataset = PotatoLeafDataset(val_samples, transform=eval_transform)
    test_dataset = PotatoLeafDataset(test_samples, transform=eval_transform)

    # Setup sampler for class imbalance
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        sampler = create_stratified_weighted_sampler(train_split_targets, num_classes=len(class_names))
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_names, class_weights
