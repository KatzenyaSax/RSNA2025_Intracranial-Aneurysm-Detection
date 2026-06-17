#!/bin/bash
#=============================================================================
# ADAM Aneurysm Segmentation — Setup & Train (for RTX 5090 32GB)
#=============================================================================
# Usage:
#   1. Clone the repo:
#      git clone https://github.com/KatzenyaSax/RSNA2025_Intracranial-Aneurysm-Detection.git
#      cd RSNA2025_Intracranial-Aneurysm-Detection
#      git checkout adam-support
#
#   2. Place ADAM dataset somewhere, e.g.:
#      /home/user/data/adamDataset/
#
#   3. Edit variables below (ADAM_DATA, nnXNet paths)
#
#   4. Run:
#      bash adam/setup_and_train.sh
#=============================================================================
set -e

# ==========================================================================
# CONFIG
# ==========================================================================
DATASET_ID=1
DATASET_NAME="Dataset001_ADAM"

# Training config
CONFIG="3d_fullres"
FOLD=0
TRAINER="nnXNetTrainer_ADAM"
NUM_EPOCHS=1000

# ==========================================================================
# 0. Environment setup
# ==========================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Auto-detect ADAM data: look for adamDataset/ next to the script
if [ -d "$SCRIPT_DIR/adamDataset" ]; then
    ADAM_DATA="$SCRIPT_DIR/adamDataset"
elif [ -d "$REPO_ROOT/adamDataset" ]; then
    ADAM_DATA="$REPO_ROOT/adamDataset"
fi

# Absolute paths inside the repo (clone-and-run, no manual config needed)
nnXNet_raw="$REPO_ROOT/adam/nnXNet_raw"
nnXNet_preprocessed="$REPO_ROOT/adam/nnXNet_preprocessed"
nnXNet_results="$REPO_ROOT/adam/nnXNet_results"

export nnXNet_raw nnXNet_preprocessed nnXNet_results

echo "============================================"
echo "ADAM Training Pipeline"
echo "============================================"
echo "Repo:        $REPO_ROOT"
echo "ADAM data:   $ADAM_DATA"
echo "nnXNet_raw:  $nnXNet_raw"
echo "nnXNet_preprocessed: $nnXNet_preprocessed"
echo "nnXNet_results: $nnXNet_results"
echo "Trainer:     $TRAINER"
echo "Config:      $CONFIG"
echo "Fold:        $FOLD"
echo "============================================"

# ==========================================================================
# 1. Install nnXNet (if not already)
# ==========================================================================
echo ""
echo "[Step 1/4] Installing nnXNet..."
cd "$REPO_ROOT"
pip install -e nnXNet --quiet
echo "  Done."

# ==========================================================================
# 2. Convert ADAM data to nnXNet format
# ==========================================================================
echo ""
echo "[Step 2/4] Converting ADAM data to nnXNet format..."
mkdir -p "$nnXNet_raw"

python adam/convert_adam_to_nnXNet.py \
    --input "$ADAM_DATA" \
    --output "$nnXNet_raw/$DATASET_NAME"
echo "  Done."

# ==========================================================================
# 3. Plan and preprocess
# ==========================================================================
echo ""
echo "[Step 3/4] Planning and preprocessing (this takes a while)..."
echo "  Extracting fingerprint & generating plans..."
nnXNet_plan_and_preprocess -d $DATASET_ID -c $CONFIG --verify_dataset_integrity

# Override batch_size to 1 in the generated plans (safety for 32GB VRAM)
PLANS_FILE="$nnXNet_preprocessed/$DATASET_NAME/nnXNetPlans.json"
if [ -f "$PLANS_FILE" ]; then
    echo "  Adjusting batch_size to 1 in plans..."
    python -c "
import json
with open('$PLANS_FILE') as f:
    plans = json.load(f)
for cfg_name in plans.get('configurations', {}):
    plans['configurations'][cfg_name]['batch_size'] = 1
with open('$PLANS_FILE', 'w') as f:
    json.dump(plans, f, indent=2)
print('  batch_size set to 1')
"
fi
echo "  Done."

# ==========================================================================
# 4. Train
# ==========================================================================
echo ""
echo "[Step 4/4] Starting training..."
echo "  Trainer: $TRAINER"
echo "  Epochs:  $NUM_EPOCHS"
echo "  Config:  $CONFIG"
echo ""

nnXNet_train $DATASET_NAME $CONFIG $FOLD -tr $TRAINER

echo ""
echo "============================================"
echo "Training complete!"
echo "Results saved to: $nnXNet_results/$DATASET_NAME"
echo "============================================"
