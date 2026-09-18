"""verify dataset integrity — counts images per class per split."""

import os
import sys


def count_images(base_path):
    """count images in each class subfolder."""
    counts = {}
    if not os.path.exists(base_path):
        print(f"  path not found: {base_path}")
        return counts
    for cls in sorted(os.listdir(base_path)):
        cls_path = os.path.join(base_path, cls)
        if os.path.isdir(cls_path):
            n = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
            counts[cls] = n
    return counts


def main():
    # resolve project root (two levels up from scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    print("=" * 60)
    print("dataset verification")
    print("=" * 60)

    # --- kaggle dataset ---
    print("\n--- kaggle brain tumor mri ---\n")
    kaggle_base = os.path.join(project_root, "data", "kaggle")

    for split in ["Training", "Testing"]:
        split_path = os.path.join(kaggle_base, split)
        counts = count_images(split_path)
        total = sum(counts.values())
        print(f"  {split}:")
        for cls, n in counts.items():
            print(f"    {cls}: {n}")
        print(f"    total: {total}")
        print()

    # --- figshare dataset ---
    print("--- figshare brain tumor mri ---\n")
    figshare_raw = os.path.join(project_root, "data", "figshare", "raw")

    if os.path.exists(figshare_raw):
        mat_files = [f for f in os.listdir(figshare_raw) if f.endswith(".mat")]
        print(f"  raw .mat files: {len(mat_files)}")
    else:
        print(f"  raw path not found: {figshare_raw}")

    # check if processed folder exists
    figshare_processed = os.path.join(project_root, "data", "figshare", "processed")
    if os.path.exists(figshare_processed):
        counts = count_images(figshare_processed)
        total = sum(counts.values())
        print(f"  processed images:")
        for cls, n in counts.items():
            print(f"    {cls}: {n}")
        print(f"    total: {total}")
    else:
        print("  processed folder: not yet created")

    print()
    print("=" * 60)
    print("verification complete")
    print("=" * 60)


if __name__ == "__main__":
    main()