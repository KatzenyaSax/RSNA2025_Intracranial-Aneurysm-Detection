"""
ADAM full-volume inference + Dice evaluation.

Usage:
  python adam/inference.py \
      --fold 0 \
      --output_dir ./adam/predictions

This loads the trained checkpoint, runs sliding-window prediction on
all validation cases, saves masks, and computes per-case Dice scores.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import nibabel as nib

# ---- Path setup ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
NNXNET_PATH = os.path.join(REPO_ROOT, "nnXNet")
sys.path.insert(0, NNXNET_PATH)

from batchgenerators.utilities.file_and_folder_operations import (
    join, load_json, maybe_mkdir_p, isfile, subdirs,
)


def find_trained_model(results_dir, dataset_name, fold):
    """Locate the trained model folder (handles nnXNet directory nesting)."""
    dataset_dir = join(results_dir, dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    # nnXNet nests results as: DatasetXXX_NAME/Trainer__Plans__Config/
    trainer_dirs = [d for d in os.listdir(dataset_dir)
                    if os.path.isdir(join(dataset_dir, d))
                    and isfile(join(dataset_dir, d, "plans.json"))]
    if not trainer_dirs:
        # Fallback: maybe the dataset_dir itself has plans.json
        if isfile(join(dataset_dir, "plans.json")):
            trainer_dirs = [""]
        else:
            raise FileNotFoundError(
                f"No model with plans.json found in {dataset_dir}")

    model_dir = join(dataset_dir, trainer_dirs[0])
    if trainer_dirs[0] != "":
        print(f"Found trainer dir: {trainer_dirs[0]}")
    fold_dir = join(model_dir, f"fold_{fold}")

    # Pick best > latest > final checkpoint
    for ckpt in ["checkpoint_best.pth", "checkpoint_latest.pth",
                 "checkpoint_final.pth"]:
        ckpt_path = join(fold_dir, ckpt)
        if isfile(ckpt_path):
            return model_dir, fold_dir, ckpt_path

    raise FileNotFoundError(f"No checkpoint found in {fold_dir}")


def load_raw_data(raw_dir, dataset_name):
    """Load raw ADAM images and labels."""
    images_dir = join(raw_dir, dataset_name, "imagesTr")
    labels_dir = join(raw_dir, dataset_name, "labelsTr")
    dataset_json = load_json(join(raw_dir, dataset_name, "dataset.json"))

    cases = []
    for fname in sorted(os.listdir(images_dir)):
        if fname.endswith("_0000.nii.gz"):
            case_id = fname.replace("_0000.nii.gz", "")
            label_path = join(labels_dir, f"{case_id}.nii.gz")
            if isfile(label_path):
                cases.append((case_id, join(images_dir, fname), label_path))

    return cases, dataset_json


def load_val_split(preprocessed_dir, dataset_name, fold):
    """Get validation case IDs from splits_final.json."""
    splits_file = join(preprocessed_dir, dataset_name, "splits_final.json")
    splits = load_json(splits_file)
    val_cases = splits[fold]["val"]
    return val_cases


def compute_dice(pred_mask, gt_mask, aneurysm_label=1):
    """Compute Dice Similarity Coefficient for a specific class."""
    pred = (pred_mask == aneurysm_label)
    gt = (gt_mask == aneurysm_label)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0  # Both empty → perfect match
    return 2.0 * intersection / total


def main():
    parser = argparse.ArgumentParser(description="ADAM inference + Dice eval")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pth")
    args = parser.parse_args()

    # Paths
    nnxnet_raw = os.environ.get("nnXNet_raw",
                                 join(REPO_ROOT, "adam", "nnXNet_raw"))
    nnxnet_preprocessed = os.environ.get(
        "nnXNet_preprocessed",
        join(REPO_ROOT, "adam", "nnXNet_preprocessed"))
    nnxnet_results = os.environ.get(
        "nnXNet_results",
        join(REPO_ROOT, "adam", "nnXNet_results"))
    dataset_name = "Dataset001_ADAM"
    fold = args.fold

    if args.output_dir is None:
        args.output_dir = join(nnxnet_results, dataset_name,
                               f"fold_{fold}", "validation_full")

    # ---- Locate model ----
    model_dir, fold_dir, checkpoint_path = find_trained_model(
        nnxnet_results, dataset_name, fold)
    print(f"Model dir:  {model_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    # ---- Patch plans.json in model output with seg_index ----
    plans_file = join(model_dir, "plans.json")
    if isfile(plans_file):
        plans = load_json(plans_file)
        patched = False
        for cfg_name, cfg in plans.get("configurations", {}).items():
            if "seg_index" not in cfg:
                cfg["seg_index"] = [[1]]
                cfg["seg_index_1"] = [[1]]
                cfg["seg_index_2"] = [[1]]
                patched = True
        if patched:
            import json as _json
            with open(plans_file, "w") as f:
                _json.dump(plans, f, indent=2)
            print("Patched plans.json with seg_index")

    # ---- Load validation split ----
    val_cases = load_val_split(nnxnet_preprocessed, dataset_name, fold)
    print(f"Validation cases ({len(val_cases)}): {val_cases[:5]}...")

    # ---- Load raw data ----
    all_cases, dataset_json = load_raw_data(nnxnet_raw, dataset_name)
    val_data = [(cid, img, lbl) for cid, img, lbl in all_cases
                if cid in val_cases]
    print(f"Matched {len(val_data)}/{len(val_cases)} validation cases with raw data")

    # ---- Initialize predictor ----
    from nnxnet.inference.predict_from_raw_data import nnXNetPredictor
    import nnxnet
    from nnxnet.utilities.find_class_by_name import recursive_find_python_class

    predictor = nnXNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_dir,
        use_folds=(fold,),
        checkpoint_name=args.checkpoint,
    )

    # ---- Monkey-patch: ResEncoderUNet_two_seg returns (seg_1, seg_2),
    #      but standard nnXNetPredictor expects a single tensor.
    #      Replace forward to return only seg_1.
    # ------------------------------------------------------------------
    _orig_forward = predictor.network.forward

    def _patched_forward(x):
        out = _orig_forward(x)
        if isinstance(out, (tuple, list)):
            return out[0]  # seg_1 only
        return out

    predictor.network.forward = _patched_forward
    print("Patched network forward for single-output inference (using seg_1)")

    # ---- Run inference and compute Dice ----
    maybe_mkdir_p(args.output_dir)
    dice_scores = []
    results = []

    for case_id, img_path, lbl_path in val_data:
        print(f"\nPredicting {case_id}...")

        # Load images as list of file paths
        from nnxnet.inference.predict_from_raw_data import (
            create_lists_from_splitted_dataset_folder,
        )

        # Predict single case
        from nnxnet.imageio.simpleitk_reader_writer import SimpleITKIO
        img, props = SimpleITKIO().read_images([img_path])
        pred = predictor.predict_single_npy_array(
            img, props, None, None, save_or_return_probabilities=False)

        # pred is a numpy array with shape [C, D, H, W] where C=2 (bg, aneurysm)
        if pred.ndim == 4 and pred.shape[0] == 2:
            pred_mask = pred.argmax(0).astype(np.int16)
        else:
            pred_mask = (pred[0] > 0.5).astype(np.int16)

        # Load ground truth
        gt_nii = nib.load(lbl_path)
        gt_mask = gt_nii.get_fdata().astype(np.int16)
        gt_mask = (gt_mask > 0).astype(np.int16)

        # Compute Dice
        dsc = compute_dice(pred_mask, gt_mask)
        dice_scores.append(dsc)
        has_aneurysm = gt_mask.sum() > 0

        # Save prediction
        pred_nii = nib.Nifti1Image(pred_mask, gt_nii.affine, gt_nii.header)
        out_path = join(args.output_dir, f"{case_id}.nii.gz")
        nib.save(pred_nii, out_path)

        results.append({
            "case_id": case_id,
            "dice": round(float(dsc), 4),
            "has_aneurysm": bool(has_aneurysm),
            "pred_path": out_path,
        })

        tag = "aneurysm" if has_aneurysm else "control"
        print(f"  {case_id}: Dice={dsc:.4f} [{tag}]")

    # ---- Summary ----
    aneurysm_dice = [r["dice"] for r in results if r["has_aneurysm"]]
    all_dice = [r["dice"] for r in results]

    print(f"\n{'='*60}")
    print(f"Fold {fold} Validation Results")
    print(f"{'='*60}")
    print(f"Total cases:        {len(results)}")
    print(f"With aneurysm:      {len(aneurysm_dice)}")
    print(f"Mean Dice (all):    {np.mean(all_dice):.4f}")
    if aneurysm_dice:
        print(f"Mean Dice (aneur):  {np.mean(aneurysm_dice):.4f}")
        print(f"Median Dice (aneur):{np.median(aneurysm_dice):.4f}")
        print(f"Max Dice:           {np.max(aneurysm_dice):.4f}")
        print(f"Min Dice:           {np.min(aneurysm_dice):.4f}")
    print(f"{'='*60}")

    # Save results JSON
    summary = {
        "fold": fold,
        "checkpoint": checkpoint_path,
        "n_cases": len(results),
        "n_aneurysm": len(aneurysm_dice),
        "mean_dice_all": float(np.mean(all_dice)),
        "mean_dice_aneurysm": float(np.mean(aneurysm_dice)) if aneurysm_dice else None,
        "median_dice_aneurysm": float(np.median(aneurysm_dice)) if aneurysm_dice else None,
        "per_case": results,
    }
    summary_path = join(args.output_dir, "dice_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
