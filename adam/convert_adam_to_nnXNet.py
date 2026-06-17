"""
Convert ADAM dataset to nnXNet raw format.

ADAM structure:
  adamDataset/
    train/data/    *.nii.gz   TOF-MRA images
    train/label/   *.nii.gz   binary aneurysm masks (0=bg, 1=aneurysm)
    test/          *.nii.gz   test images (no labels)

nnXNet raw format:
  nnXNet_raw/Dataset001_ADAM/
    imagesTr/   case_XX_0000.nii.gz
    labelsTr/   case_XX.nii.gz
    dataset.json

Usage:
  python adam/convert_adam_to_nnXNet.py \
      --input ./adamDataset \
      --output $nnXNet_raw/Dataset001_ADAM
"""

import os
import sys
import argparse
import json
import shutil
import numpy as np
import nibabel as nib


def reorient_to_lps(nii_img):
    """Reorient NIfTI to LPS+ coordinate system (standard for nnXNet)."""
    import nibabel.orientations as nio
    target = "LPS"
    current = "".join(nib.aff2axcodes(nii_img.affine))
    if current == target:
        return nii_img
    orig_ornt = nio.io_orientation(nii_img.affine)
    targ_ornt = nio.axcodes2ornt(target)
    transform = nio.ornt_transform(orig_ornt, targ_ornt)
    return nii_img.as_reoriented(transform)


def convert_case(nii_path, output_path, is_label=False):
    """
    Load a .nii.gz, reorient to LPS, save.
    For labels: ensure integer type, binarize (0/1).
    """
    img = nib.load(nii_path)

    # Reorient
    img = reorient_to_lps(img)

    data = img.get_fdata()

    if is_label:
        data = (data > 0).astype(np.int16)
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        new_img.header.set_data_dtype(np.int16)
    else:
        data = data.astype(np.float32)
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        new_img.header.set_data_dtype(np.float32)

    nib.save(new_img, output_path)
    return img.shape


def main():
    parser = argparse.ArgumentParser(
        description="Convert ADAM dataset to nnXNet raw format")
    parser.add_argument("--input", required=True,
                        help="Path to ADAM dataset root (contains train/data, train/label)")
    parser.add_argument("--output", required=True,
                        help="Output directory for nnXNet raw dataset")
    parser.add_argument("--num_cases", type=int, default=None,
                        help="Limit number of training cases (default: all)")
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    # Validate input
    train_data_dir = os.path.join(input_dir, "train", "data")
    train_label_dir = os.path.join(input_dir, "train", "label")
    if not os.path.isdir(train_data_dir):
        print(f"[ERROR] Training data directory not found: {train_data_dir}")
        sys.exit(1)
    if not os.path.isdir(train_label_dir):
        print(f"[ERROR] Training label directory not found: {train_label_dir}")
        sys.exit(1)

    # Create output directories
    images_tr_dir = os.path.join(output_dir, "imagesTr")
    labels_tr_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)

    # Discover cases (files that exist in BOTH data and label)
    data_files = sorted(os.listdir(train_data_dir))
    label_files = set(os.listdir(train_label_dir))

    cases = []
    skipped = []
    for fname in data_files:
        if not fname.endswith(".nii.gz"):
            continue
        case_id = fname.replace(".nii.gz", "")
        if fname in label_files:
            cases.append(case_id)
        else:
            skipped.append(case_id)

    if args.num_cases is not None:
        cases = cases[:args.num_cases]

    print(f"Found {len(cases)} cases with matching labels")
    if skipped:
        print(f"Skipped {len(skipped)} cases without labels: {skipped[:5]}...")

    # Convert each case
    for case_id in cases:
        src_data = os.path.join(train_data_dir, f"{case_id}.nii.gz")
        src_label = os.path.join(train_label_dir, f"{case_id}.nii.gz")

        # nnXNet naming: case_XX_0000.nii.gz (0000 = channel 0)
        dst_data = os.path.join(images_tr_dir, f"{case_id}_0000.nii.gz")
        dst_label = os.path.join(labels_tr_dir, f"{case_id}.nii.gz")

        shape = convert_case(src_data, dst_data, is_label=False)
        lbl_shape = convert_case(src_label, dst_label, is_label=True)

        n_pos = nib.load(dst_label).get_fdata().sum()
        tag = "aneurysm" if n_pos > 0 else "control"
        print(f"  {case_id}: image={list(shape)} -> {tag} ({int(n_pos)} voxels)")

    # Write dataset.json
    dataset_json = {
        "name": "Dataset001_ADAM",
        "description": "ADAM: Aneurysm Detection And segMentation challenge. "
                       "TOF-MRA binary aneurysm segmentation.",
        "reference": "https://adam.isi.uu.nl/",
        "licence": "see ADAM challenge terms",
        "release": "1.0",
        "channel_names": {
            "0": "TOF-MRA",
        },
        "labels": {
            "background": 0,
            "aneurysm": 1,
        },
        "regions_class_order": [1, 2],
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n[OK] Converted {len(cases)} cases to {output_dir}")
    print(f"  imagesTr: {len(os.listdir(images_tr_dir))} files")
    print(f"  labelsTr: {len(os.listdir(labels_tr_dir))} files")
    print(f"  dataset.json written")

    # Print next steps
    print(f"\nNext steps:")
    print(f"  export nnXNet_raw='{os.path.dirname(output_dir)}'")
    print(f"  export nnXNet_preprocessed='/path/to/nnXNet_preprocessed'")
    print(f"  export nnXNet_results='/path/to/nnXNet_results'")
    print(f"  nnXNet_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity")
    print(f"  nnXNet_train Dataset001_ADAM 3d_fullres 0 -tr nnXNetTrainer_ADAM")


if __name__ == "__main__":
    main()
