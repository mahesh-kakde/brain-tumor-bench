"""convert figshare .mat files to png images, organized by class."""

import os
import sys
import numpy as np
import h5py
from PIL import Image

# label mapping from the figshare readme
LABEL_MAP = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}


def convert_mat_to_png(mat_path):
    """read a .mat file and return (image_array_uint8, class_name)."""
    with h5py.File(mat_path, "r") as f:
        cjdata = f["cjdata"]
        label = int(cjdata["label"][0, 0])
        image = np.array(cjdata["image"], dtype=np.float64)

    # normalize to 0-255
    min_val = image.min()
    max_val = image.max()
    if max_val > min_val:
        image = (255.0 / (max_val - min_val)) * (image - min_val)
    image = image.astype(np.uint8)

    class_name = LABEL_MAP.get(label)
    if class_name is None:
        print(f"  warning: unknown label {label} in {mat_path}")
        return None, None

    return image, class_name


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    raw_dir = os.path.join(project_root, "data", "figshare", "raw")
    out_dir = os.path.join(project_root, "data", "figshare", "processed")

    if not os.path.exists(raw_dir):
        print(f"error: raw directory not found: {raw_dir}")
        sys.exit(1)

    # collect all .mat files
    mat_files = sorted(
        [f for f in os.listdir(raw_dir) if f.endswith(".mat")],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    print(f"found {len(mat_files)} .mat files")

    # create output directories
    for cls in LABEL_MAP.values():
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    # convert
    counts = {cls: 0 for cls in LABEL_MAP.values()}
    errors = 0

    for i, fname in enumerate(mat_files):
        mat_path = os.path.join(raw_dir, fname)
        try:
            image, class_name = convert_mat_to_png(mat_path)
            if image is None:
                errors += 1
                continue

            # save as png
            base = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, class_name, f"{base}.png")
            Image.fromarray(image).save(out_path)
            counts[class_name] += 1
        except Exception as e:
            print(f"  error processing {fname}: {e}")
            errors += 1

        # progress update every 500 files
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(mat_files)}...")

    # summary
    print()
    print("=" * 50)
    print("conversion complete")
    print("=" * 50)
    total = sum(counts.values())
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")
    print(f"  total: {total}")
    if errors > 0:
        print(f"  errors: {errors}")
    print(f"  output: {out_dir}")


if __name__ == "__main__":
    main()