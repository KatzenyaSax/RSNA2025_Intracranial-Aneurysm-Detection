"""
Evaluate trained model on holdout set — compute full-volume Dice.

The holdout cases live in nnXNet_raw/Dataset001_ADAM/imagesTs/ and labelsTs/.
They were NEVER seen during training or nnXNet's internal cross-validation.

Usage:
  python adam/eval_holdout.py --fold 0
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import nibabel as nib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
NNXNET_PATH = os.path.join(REPO_ROOT, "nnXNet")
sys.path.insert(0, NNXNET_PATH)

from batchgenerators.utilities.file_and_folder_operations import (
    join, load_json, maybe_mkdir_p, isfile,
)


def find_model(results_dir, dataset_name, fold):
    """Locate the trained model folder."""
    dataset_dir = join(results_dir, dataset_name)
    trainer_dirs = [d for d in os.listdir(dataset_dir)
                    if os.path.isdir(join(dataset_dir, d))
                    and isfile(join(dataset_dir, d, "plans.json"))]
    if not trainer_dirs:
        raise FileNotFoundError(f"No model with plans.json in {dataset_dir}")
    model_dir = join(dataset_dir, trainer_dirs[0])
    fold_dir = join(model_dir, f"fold_{fold}")
    for ckpt in ["checkpoint_best.pth", "checkpoint_latest.pth",
                 "checkpoint_final.pth"]:
        ckpt_path = join(fold_dir, ckpt)
        if isfile(ckpt_path):
            return model_dir, ckpt_path
    raise FileNotFoundError(f"No checkpoint in {fold_dir}")


def collect_holdout(raw_dir, dataset_name):
    """Find all cases in imagesTs/ with matching labelsTs/."""
    images_dir = join(raw_dir, dataset_name, "imagesTs")
    labels_dir = join(raw_dir, dataset_name, "labelsTs")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No holdout images: {images_dir}")

    cases = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.endswith("_0000.nii.gz"):
            continue
        case_id = fname.replace("_0000.nii.gz", "")
        lbl_path = join(labels_dir, f"{case_id}.nii.gz")
        if isfile(lbl_path):
            cases.append((case_id, join(images_dir, fname), lbl_path))
    return cases


def compute_dice(pred_mask, gt_mask):
    """Dice for the positive class."""
    pred = (pred_mask == 1)
    gt = (gt_mask == 1)
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection / total)


def compute_detection_metrics(pred_mask, gt_mask):
    """Compute detection sensitivity and FP count."""
    has_pred = int(pred_mask.sum() > 0)
    has_gt = int(gt_mask.sum() > 0)
    # Per-case: TP, FP, FN, TN
    tp = int(has_pred == 1 and has_gt == 1)
    fp = int(has_pred == 1 and has_gt == 0)
    fn = int(has_pred == 0 and has_gt == 1)
    tn = int(has_pred == 0 and has_gt == 0)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def shape_align(pred_mask, gt_mask):
    """Align prediction shape to ground truth (handles SimpleITK vs nibabel)."""
    if pred_mask.shape == gt_mask.shape:
        return pred_mask
    # Try transpositions
    for perm in [(1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)]:
        t = pred_mask.transpose(perm)
        if t.shape == gt_mask.shape:
            return t
    # Last resort: brute force
    if pred_mask.shape == gt_mask.T.shape:
        return pred_mask.T
    return pred_mask  # return as-is, let caller handle


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate on held-out test set")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    nnxnet_raw = os.environ.get(
        "nnXNet_raw", join(REPO_ROOT, "adam", "nnXNet_raw"))
    nnxnet_results = os.environ.get(
        "nnXNet_results", join(REPO_ROOT, "adam", "nnXNet_results"))
    dataset_name = "Dataset001_ADAM"
    fold = args.fold

    # Locate model
    model_dir, checkpoint_path = find_model(
        nnxnet_results, dataset_name, fold)
    print(f"Model:  {model_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    # Collect holdout cases
    holdout_cases = collect_holdout(nnxnet_raw, dataset_name)
    print(f"Holdout cases: {len(holdout_cases)}")

    # Patch plans
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
            with open(plans_file, "w") as f:
                json.dump(plans, f, indent=2)
            print("Patched plans.json")

    # Initialize predictor
    from nnxnet.inference.predict_from_raw_data import nnXNetPredictor
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
        checkpoint_name=os.path.basename(checkpoint_path),
    )

    # Monkey-patch forward for dual-output model
    _orig_forward = predictor.network.forward

    def _patched_forward(x):
        out = _orig_forward(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    predictor.network.forward = _patched_forward

    # Output directory
    output_dir = join(model_dir, "holdout_eval")
    maybe_mkdir_p(output_dir)

    # Run inference
    from nnxnet.imageio.simpleitk_reader_writer import SimpleITKIO

    results = []
    dice_scores = []
    aneurysm_dice = []
    detections = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}

    for case_id, img_path, lbl_path in holdout_cases:
        print(f"\nPredicting {case_id}...")

        img, props = SimpleITKIO().read_images([img_path])
        pred = predictor.predict_single_npy_array(
            img, props, None, None, save_or_return_probabilities=False)

        if pred.ndim == 4:
            pred_mask = pred.argmax(0).astype(np.int16)
        else:
            pred_mask = (pred > 0.5).astype(np.int16)

        gt_nii = nib.load(lbl_path)
        gt_mask = gt_nii.get_fdata().astype(np.int16)
        gt_mask = (gt_mask > 0).astype(np.int16)

        print(f"  Pred shape: {pred_mask.shape}, GT shape: {gt_mask.shape}")

        # Align orientations
        pred_mask = shape_align(pred_mask, gt_mask)
        if pred_mask.shape != gt_mask.shape:
            print(f"  WARNING: shape mismatch after alignment, skipping")
            continue

        # Metrics
        dsc = compute_dice(pred_mask, gt_mask)
        dice_scores.append(dsc)
        has_gt = gt_mask.sum() > 0

        det = compute_detection_metrics(pred_mask, gt_mask)
        for k in det:
            detections[k] += det[k]

        tag = "aneurysm" if has_gt else "control"
        print(f"  Dice={dsc:.4f} [{tag}]")

        if has_gt:
            aneurysm_dice.append(dsc)

        # Save prediction
        pred_nii = nib.Nifti1Image(pred_mask, gt_nii.affine, gt_nii.header)
        out_path = join(output_dir, f"{case_id}.nii.gz")
        nib.save(pred_nii, out_path)

        results.append({
            "case_id": case_id,
            "dice": round(dsc, 4),
            "has_aneurysm": bool(has_gt),
            "pred_path": out_path,
        })

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"Holdout Evaluation Results (fold {fold})")
    print(f"{'='*60}")
    print(f"Total cases:           {len(results)}")
    print(f"With aneurysm:         {len(aneurysm_dice)}")
    print(f"Mean Dice (all):       {np.mean(dice_scores):.4f}")
    if aneurysm_dice:
        print(f"Mean Dice (aneurysm):  {np.mean(aneurysm_dice):.4f}")
        print(f"Median Dice (aneurysm):{np.median(aneurysm_dice):.4f}")
        print(f"Max Dice:              {np.max(aneurysm_dice):.4f}")
    print(f"---")
    n_total = len(results)
    tp, fp, fn, tn = detections["TP"], detections["FP"], detections["FN"], detections["TN"]
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fp_per_case = fp / n_total if n_total > 0 else 0
    print(f"Detection sensitivity: {sensitivity:.4f}")
    print(f"Detection specificity: {specificity:.4f}")
    print(f"FP per case:           {fp_per_case:.2f}")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "fold": fold,
        "checkpoint": checkpoint_path,
        "n_cases": len(results),
        "n_aneurysm": len(aneurysm_dice),
        "mean_dice_all": float(np.mean(dice_scores)),
        "mean_dice_aneurysm": float(np.mean(aneurysm_dice)) if aneurysm_dice else None,
        "median_dice_aneurysm": float(np.median(aneurysm_dice)) if aneurysm_dice else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "fp_per_case": fp_per_case,
        "detection_counts": detections,
        "per_case": results,
    }
    summary_path = join(output_dir, "holdout_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
