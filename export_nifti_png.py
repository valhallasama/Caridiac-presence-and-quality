import argparse
import os

import cv2
import nibabel as nib
import numpy as np


def export_nifti_to_png(nifti_path: str, out_dir: str | None = None) -> None:
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"Input NIfTI not found: {nifti_path}")

    if out_dir is None:
        base = os.path.splitext(os.path.basename(nifti_path))[0]
        # strip an extra .gz if present
        if base.endswith(".nii"):
            base = base[:-4]
        out_dir = os.path.join(os.path.dirname(nifti_path), base + "_png")

    os.makedirs(out_dir, exist_ok=True)

    nii = nib.load(nifti_path)
    arr = nii.get_fdata()

    # Ensure we always have a slice dimension
    if arr.ndim == 2:
        arr = arr[..., None]

    h, w, num_slices = arr.shape[0], arr.shape[1], arr.shape[-1]
    print(f"Input shape: {arr.shape} -> saving {num_slices} slice(s) to {out_dir}")

    for i in range(num_slices):
        sl = arr[..., i].astype(np.float32)
        sl -= sl.min()
        maxv = sl.max()
        if maxv > 0:
            sl /= maxv
        sl_u8 = (sl * 255.0).clip(0, 255).astype(np.uint8)
        out_path = os.path.join(out_dir, f"slice_{i:03d}.png")
        cv2.imwrite(out_path, sl_u8)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NIfTI volume to a folder of PNG slices.")
    parser.add_argument("nifti_path", type=str, help="Path to .nii or .nii.gz file")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: <nifti_basename>_png beside input)")
    args = parser.parse_args()

    export_nifti_to_png(args.nifti_path, args.out_dir)


if __name__ == "__main__":
    main()
