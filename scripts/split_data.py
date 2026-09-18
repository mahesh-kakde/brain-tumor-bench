"""create stratified train/val split from kaggle training data."""

import os
import json
from sklearn.model_selection import train_test_split

# fixed seed for reproducibility across all experiments
SPLIT_SEED = 42
VAL_RATIO = 0.15

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    train_base = os.path.join(project_root, "data", "kaggle", "Training")
    out_dir = os.path.join(project_root, "data", "splits")
    os.makedirs(out_dir, exist_ok=True)

    all_files = []
    all_labels = []

    # collect all training image paths (relative to Training/)
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(train_base, cls)
        if not os.path.isdir(cls_dir):
            print(f"error: class directory not found: {cls_dir}")
            return
        files = sorted(os.listdir(cls_dir))
        for f in files:
            # store as "class/filename" for portability
            all_files.append(f"{cls}/{f}")
            all_labels.append(cls)

    print(f"total training images: {len(all_files)}")

    # stratified split
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files,
        all_labels,
        test_size=VAL_RATIO,
        random_state=SPLIT_SEED,
        stratify=all_labels,
    )

    # count per class
    print(f"\ntrain subset: {len(train_files)}")
    train_counts = {}
    for label in train_labels:
        train_counts[label] = train_counts.get(label, 0) + 1
    for cls in CLASS_NAMES:
        print(f"  {cls}: {train_counts.get(cls, 0)}")

    print(f"\nval subset: {len(val_files)}")
    val_counts = {}
    for label in val_labels:
        val_counts[label] = val_counts.get(label, 0) + 1
    for cls in CLASS_NAMES:
        print(f"  {cls}: {val_counts.get(cls, 0)}")

    # save split to json
    split_data = {
        "seed": SPLIT_SEED,
        "val_ratio": VAL_RATIO,
        "train_count": len(train_files),
        "val_count": len(val_files),
        "train_per_class": train_counts,
        "val_per_class": val_counts,
        "train": sorted(train_files),
        "val": sorted(val_files),
    }

    out_path = os.path.join(out_dir, "train_val_split.json")
    with open(out_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"\nsplit saved to: {out_path}")


if __name__ == "__main__":
    main()