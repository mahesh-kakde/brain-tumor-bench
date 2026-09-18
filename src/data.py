# dataset and dataloader utilities
import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# imagenet normalization stats (required for pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class KaggleTumorDataset(Dataset):
    """loads kaggle brain tumor images using a pre-computed split json."""

    def __init__(self, kaggle_train_dir, split_file, subset, transform=None):
        """
        args:
            kaggle_train_dir: path to data/kaggle/Training/
            split_file: path to data/splits/train_val_split.json
            subset: 'train' or 'val'
            transform: torchvision transform to apply
        """
        assert subset in ("train", "val"), "subset must be 'train' or 'val'"
        with open(split_file, "r") as f:
            split = json.load(f)

        self.transform = transform
        self.samples = []  # list of (image_path, label_idx)

        for rel_path in split[subset]:
            # rel_path is "class/filename"
            cls_name = rel_path.split("/")[0]
            img_path = os.path.join(kaggle_train_dir, rel_path.replace("/", os.sep))
            self.samples.append((img_path, CLASS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class KaggleTestDataset(Dataset):
    """loads the held-out kaggle test set."""

    def __init__(self, kaggle_test_dir, transform=None):
        """
        args:
            kaggle_test_dir: path to data/kaggle/Testing/
            transform: torchvision transform to apply
        """
        self.transform = transform
        self.samples = []

        for cls_name in CLASS_NAMES:
            cls_dir = os.path.join(kaggle_test_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                img_path = os.path.join(cls_dir, fname)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, CLASS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class FigshareDataset(Dataset):
    """loads figshare processed images (3 classes, sensitivity check only)."""

    FIGSHARE_CLASSES = ["glioma", "meningioma", "pituitary"]
    FIGSHARE_CLASS_TO_IDX = {c: i for i, c in enumerate(FIGSHARE_CLASSES)}

    def __init__(self, figshare_processed_dir, transform=None):
        self.transform = transform
        self.samples = []

        for cls_name in self.FIGSHARE_CLASSES:
            cls_dir = os.path.join(figshare_processed_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                img_path = os.path.join(cls_dir, fname)
                if os.path.isfile(img_path):
                    self.samples.append((img_path, self.FIGSHARE_CLASS_TO_IDX[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(kaggle_train_dir, kaggle_test_dir, split_file,
                    batch_size=32, num_workers=2):
    """build train, val, and test dataloaders."""
    import torch
    pin = torch.cuda.is_available()

    train_ds = KaggleTumorDataset(kaggle_train_dir, split_file, "train",
                                  transform=get_train_transforms())
    val_ds   = KaggleTumorDataset(kaggle_train_dir, split_file, "val",
                                  transform=get_val_test_transforms())
    test_ds  = KaggleTestDataset(kaggle_test_dir,
                                  transform=get_val_test_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader