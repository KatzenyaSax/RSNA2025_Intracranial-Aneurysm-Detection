#!/bin/bash
#=============================================================================
# ADAM Training Only — reuses preprocessed data from setup_and_train.sh
#=============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

export nnXNet_raw="$REPO_ROOT/adam/nnXNet_raw"
export nnXNet_preprocessed="$REPO_ROOT/adam/nnXNet_preprocessed"
export nnXNet_results="$REPO_ROOT/adam/nnXNet_results"

DATASET_NAME="Dataset001_ADAM"
CONFIG="3d_fullres"
FOLD=0
TRAINER="nnXNetTrainer_ADAM"

echo "============================================"
echo "ADAM Training"
echo "============================================"
echo "Trainer:     $TRAINER"
echo "Config:      $CONFIG"
echo "Fold:        $FOLD"
echo "============================================"

nnXNet_train $DATASET_NAME $CONFIG $FOLD -tr $TRAINER
