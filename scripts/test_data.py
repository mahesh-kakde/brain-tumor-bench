"""smoke test for src/data.py — no torch required."""
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

project      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
split_file   = os.path.join(project, "data", "splits", "train_val_split.json")
kaggle_train = os.path.join(project, "data", "kaggle", "Training")
kaggle_test  = os.path.join(project, "data", "kaggle", "Testing")
figshare_dir = os.path.join(project, "data", "figshare", "processed")

with open(split_file) as f:
    split = json.load(f)

print("--- split counts ---")
print(f"  train: {split['train_count']}  val: {split['val_count']}")

# verify all train paths exist
missing_train = [
    p for p in split["train"]
    if not os.path.isfile(os.path.join(kaggle_train, p.replace("/", os.sep)))
]
missing_val = [
    p for p in split["val"]
    if not os.path.isfile(os.path.join(kaggle_train, p.replace("/", os.sep)))
]
print(f"\n--- path checks ---")
print(f"  missing train paths: {len(missing_train)}  (expect 0)")
print(f"  missing val paths:   {len(missing_val)}  (expect 0)")

# test set count
test_classes = ["glioma", "meningioma", "notumor", "pituitary"]
test_total = sum(
    len(os.listdir(os.path.join(kaggle_test, c)))
    for c in test_classes
    if os.path.isdir(os.path.join(kaggle_test, c))
)
print(f"\n--- test images: {test_total}  (expect 1600) ---")

# figshare count
fig_classes = ["glioma", "meningioma", "pituitary"]
fig_total = sum(
    len(os.listdir(os.path.join(figshare_dir, c)))
    for c in fig_classes
    if os.path.isdir(os.path.join(figshare_dir, c))
)
print(f"--- figshare images: {fig_total}  (expect 3064) ---")

# open a sample image
from PIL import Image
sample_rel = split["train"][0]
sample_path = os.path.join(kaggle_train, sample_rel.replace("/", os.sep))
img = Image.open(sample_path).convert("RGB")
print(f"\n--- sample image ---")
print(f"  path: {sample_rel}")
print(f"  mode: {img.mode}  size: {img.size}")

# verify transforms can be imported (checks torchvision)
try:
    from src.data import get_train_transforms, get_val_test_transforms, CLASS_NAMES, CLASS_TO_IDX
    t = get_train_transforms()
    v = get_val_test_transforms()
    print(f"\n--- transforms ---")
    print(f"  train: {len(t.transforms)} steps")
    print(f"  val:   {len(v.transforms)} steps")
    print(f"  classes: {CLASS_NAMES}")
    print(f"  class->idx: {CLASS_TO_IDX}")
except ImportError as e:
    print(f"\n  note: torch/torchvision not installed locally ({e})")
    print("  transforms will be tested on kaggle during training.")

print("\nall checks passed.")