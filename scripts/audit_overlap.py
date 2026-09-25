"""provenance and overlap audit between kaggle and figshare datasets.

checks:
  1. sha-256 exact duplicate detection (expect 0)
  2. perceptual near-duplicate detection via phash (expect ~706 pairs)

output saved to results/audit_log.json.
"""

import os
import sys
import json
import hashlib
from collections import defaultdict

import imagehash
from PIL import Image

# perceptual hash distance threshold — images with distance <= this are near-duplicates
PHASH_THRESHOLD = 10

KAGGLE_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
FIGSHARE_CLASSES = ["glioma", "meningioma", "pituitary"]


def sha256_file(path):
    """compute sha-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_images(base_dir, classes):
    """return list of (relative_path, full_path) for all images."""
    images = []
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            full = os.path.join(cls_dir, fname)
            if os.path.isfile(full):
                images.append((f"{cls}/{fname}", full))
    return images


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    kaggle_train_dir = os.path.join(project_root, "data", "kaggle", "Training")
    figshare_dir = os.path.join(project_root, "data", "figshare", "processed")
    out_dir = os.path.join(project_root, "results")
    os.makedirs(out_dir, exist_ok=True)

    print("collecting image paths...")
    kaggle_images = collect_images(kaggle_train_dir, KAGGLE_CLASSES)
    figshare_images = collect_images(figshare_dir, FIGSHARE_CLASSES)
    print(f"  kaggle train: {len(kaggle_images)} images")
    print(f"  figshare:     {len(figshare_images)} images")

    # --- step 1: sha-256 exact duplicate check ---
    print("\nstep 1: sha-256 exact duplicate check...")

    # build hash → list of paths for figshare
    figshare_hashes = defaultdict(list)
    for i, (rel, full) in enumerate(figshare_images):
        figshare_hashes[sha256_file(full)].append(rel)
        if (i + 1) % 500 == 0:
            print(f"  hashed {i + 1}/{len(figshare_images)} figshare images...")

    exact_matches = []
    for i, (rel, full) in enumerate(kaggle_images):
        h = sha256_file(full)
        if h in figshare_hashes:
            for fig_rel in figshare_hashes[h]:
                exact_matches.append({"kaggle": rel, "figshare": fig_rel, "sha256": h})
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(kaggle_images)} kaggle images...")

    print(f"  exact sha-256 matches: {len(exact_matches)}")

    # --- step 2: perceptual near-duplicate check (phash) ---
    print("\nstep 2: perceptual near-duplicate check (phash)...")
    print(f"  threshold: hamming distance <= {PHASH_THRESHOLD}")

    # compute phash for all figshare images
    figshare_phashes = []
    for i, (rel, full) in enumerate(figshare_images):
        try:
            ph = imagehash.phash(Image.open(full).convert("L"))
            figshare_phashes.append((rel, ph))
        except Exception as e:
            print(f"  warning: could not hash {rel}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  phashed {i + 1}/{len(figshare_images)} figshare images...")

    # compare each kaggle image against all figshare phashes
    near_duplicate_pairs = []
    for i, (k_rel, k_full) in enumerate(kaggle_images):
        try:
            k_ph = imagehash.phash(Image.open(k_full).convert("L"))
        except Exception:
            continue
        for f_rel, f_ph in figshare_phashes:
            dist = k_ph - f_ph
            if dist <= PHASH_THRESHOLD:
                near_duplicate_pairs.append({
                    "kaggle": k_rel,
                    "figshare": f_rel,
                    "phash_distance": int(dist),
                })
        if (i + 1) % 500 == 0:
            print(f"  compared {i + 1}/{len(kaggle_images)} kaggle images...")

    print(f"  near-duplicate pairs found: {len(near_duplicate_pairs)}")

    # --- save results ---
    audit_log = {
        "kaggle_train_images": len(kaggle_images),
        "figshare_images": len(figshare_images),
        "phash_threshold": PHASH_THRESHOLD,
        "exact_sha256_matches": len(exact_matches),
        "near_duplicate_pairs": len(near_duplicate_pairs),
        "exact_matches_detail": exact_matches,
        "near_duplicates_detail": near_duplicate_pairs[:50],  # first 50 for inspection
    }

    out_path = os.path.join(out_dir, "audit_log.json")
    with open(out_path, "w") as f:
        json.dump(audit_log, f, indent=2)

    print(f"\naudit log saved to: {out_path}")
    print("\n--- summary ---")
    print(f"  exact duplicates (sha-256): {len(exact_matches)}")
    print(f"  near-duplicates (phash):    {len(near_duplicate_pairs)}")


if __name__ == "__main__":
    main()