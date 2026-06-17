"""
Convert ADAM dataset to nnXNet raw format with optional holdout split.

ADAM structure:
  adamDataset/
    train/data/    *.nii.gz   TOF-MRA images
    train/label/   *.nii.gz   binary aneurysm masks (0=bg, 1=aneurysm)
    test/          *.nii.gz   test images (no labels)

nnXNet raw format:
  nnXNet_raw/Dataset001_ADAM/
    imagesTr/   case_XX_0000.nii.gz   (training images)
    labelsTr/   case_XX.nii.gz         (training labels)
    imagesTs/   case_XX_0000.nii.gz   (holdout images, if --holdout_count > 0)
    labelsTs/   case_XX.nii.gz         (holdout labels)
    dataset.json

Usage:
  # No holdout (all data in training)
  python adam/convert_adam_to_nnXNet.py -i ./adamDataset -o $nnXNet_raw/Dataset001_ADAM

  # With 18-case holdout
  python adam/convert_adam_to_nnXNet.py -i ./adamDataset -o $nnXNet_raw/Dataset001_ADAM \
      --holdout_count 18 --seed 42
"""

import os
import sys
import argparse
import json
import random
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
    """Load a .nii.gz, reorient to LPS, save."""
    img = nib.load(nii_path)
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
    parser.add_argument("--input", "-i", required=True,
                        help="Path to ADAM dataset root (contains train/data, train/label)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for nnXNet raw dataset")
    parser.add_argument("--holdout_count", type=int, default=0,
                        help="Number of cases to hold out for testing (default: 0 = all training)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for holdout split (default: 42)")
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

    # Discover cases
    data_files = sorted(os.listdir(train_data_dir))
    label_files = set(os.listdir(train_label_dir))

    all_cases = []
    skipped = []
    for fname in data_files:
        if not fname.endswith(".nii.gz"):
            continue
        case_id = fname.replace(".nii.gz", "")
        if fname in label_files:
            all_cases.append(case_id)
        else:
            skipped.append(case_id)

    print(f"Found {len(all_cases)} cases with matching labels")
    if skipped:
        print(f"Skipped {len(skipped)} cases without labels: {skipped[:5]}...")

    # ---- Holdout split ----
    random.seed(args.seed)
    holdout_cases = set()
    if args.holdout_count > 0:
        # Separate aneurysm and control for stratified split
        aneurysm_cases = []
        control_cases = []
        for cid in all_cases:
            lbl_path = os.path.join(train_label_dir, f"{cid}.nii.gz")
            lbl = nib.load(lbl_path).get_fdata()
            if lbl.sum() > 0:
                aneurysm_cases.append(cid)
            else:
                control_cases.append(cid)

        # Hold out proportionally: ~70% aneurysm, ~30% control
        n_aneurysm = int(round(args.holdout_count * 0.7))
        n_control = args.holdout_count - n_aneurysm
        n_aneurysm = min(n_aneurysm, len(aneurysm_cases))
        n_control = min(n_control, len(control_cases))

        random.shuffle(aneurysm_cases)
        random.shuffle(control_cases)

        holdout_cases = set(
            aneurysm_cases[:n_aneurysm] + control_cases[:n_control]
        )
        print(f"\nHoldout split (seed={args.seed}):")
        print(f"  Aneurysm: {n_aneurysm}/{len(aneurysm_cases)}")
        print(f"  Control:  {n_control}/{len(control_cases)}")
        print(f"  Total holdout: {len(holdout_cases)}")
        print(f"  Training: {len(all_cases) - len(holdout_cases)}")

    train_cases = [c for c in all_cases if c not in holdout_cases]
    holdout_list = sorted(holdout_cases)

    # Create output directories
    images_tr_dir = os.path.join(output_dir, "imagesTr")
    labels_tr_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)

    if args.holdout_count > 0:
        images_ts_dir = os.path.join(output_dir, "imagesTs")
        labels_ts_dir = os.path.join(output_dir, "labelsTs")
        os.makedirs(images_ts_dir, exist_ok=True)
        os.makedirs(labels_ts_dir, exist_ok=True)

    # ---- Convert training cases ----
    print(f"\nConverting {len(train_cases)} training cases...")
    for case_id in train_cases:
        src_data = os.path.join(train_data_dir, f"{case_id}.nii.gz")
        src_label = os.path.join(train_label_dir, f"{case_id}.nii.gz")

        dst_data = os.path.join(images_tr_dir, f"{case_id}_0000.nii.gz")
        dst_label = os.path.join(labels_tr_dir, f"{case_id}.nii.gz")

        convert_case(src_data, dst_data, is_label=False)
        convert_case(src_label, dst_label, is_label=True)

        n_pos = nib.load(dst_label).get_fdata().sum()
        tag = "aneurysm" if n_pos > 0 else "control"
        print(f"  [train] {case_id}: -> {tag} ({int(n_pos)} voxels)")

    # ---- Convert holdout cases ----
    if args.holdout_count > 0:
        print(f"\nConverting {len(holdout_list)} holdout cases...")
        for case_id in holdout_list:
            src_data = os.path.join(train_data_dir, f"{case_id}.nii.gz")
            src_label = os.path.join(train_label_dir, f"{case_id}.nii.gz")

            dst_data = os.path.join(images_ts_dir, f"{case_id}_0000.nii.gz")
            dst_label = os.path.join(labels_ts_dir, f"{case_id}.nii.gz")

            convert_case(src_data, dst_data, is_label=False)
            convert_case(src_label, dst_label, is_label=True)

            n_pos = nib.load(dst_label).get_fdata().sum()
            tag = "aneurysm" if n_pos > 0 else "control"
            print(f"  [holdout] {case_id}: -> {tag} ({int(n_pos)} voxels)")

        # Save holdout case list for reference
        holdout_file = os.path.join(output_dir, "holdout_cases.json")
        with open(holdout_file, "w") as f:
            json.dump({
                "seed": args.seed,
                "holdout_count": args.holdout_count,
                "cases": holdout_list,
            }, f, indent=2)

    # ---- Write dataset.json ----
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
        "numTraining": len(train_cases),
        "file_ending": ".nii.gz",
    }

    json_path = os.path.join(output_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n[OK] Converted {len(train_cases)} train + {len(holdout_list)} holdout cases to {output_dir}")
    print(f"  imagesTr: {len(os.listdir(images_tr_dir))} files")
    print(f"  labelsTr: {len(os.listdir(labels_tr_dir))} files")
    if args.holdout_count > 0:
        print(f"  imagesTs: {len(os.listdir(images_ts_dir))} files")
        print(f"  labelsTs: {len(os.listdir(labels_ts_dir))} files")

    print(f"\nNext steps:")
    print(f"  nnXNet_plan_and_preprocess -d 1 -c 3d_fullres")
    print(f"  nnXNet_train Dataset001_ADAM 3d_fullres 0 -tr nnXNetTrainer_ADAM")
    if args.holdout_count > 0:
        print(f"  python adam/eval_holdout.py --fold 0")


if __name__ == "__main__":
    main()
