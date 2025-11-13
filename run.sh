#!/bin/bash

# Basic training script for multitab codebase
# Modify these parameters as needed

DATA_ROOT="/path/to/data/"  # Set your data root directory here
MODEL_NAME="mtt" # Name of the model to use (MultiTab-Net = mtt)
DATASET="acs_income" # Dataset to use, e.g., 'acs_income' or 'higgs' or 'aliexpress' or 'synthetic_v2'
GPU_ID=0 # GPU ID to use for training
SEED=42 # Random seed for reproducibility
PATIENCE=5 # Patience for early stopping

echo "Starting training with:"
echo "  Data Root: $DATA_ROOT"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET"
echo "  GPU: $GPU_ID"
echo "  Seed: $SEED"
echo "  Patience: $PATIENCE"

python main.py --model $MODEL_NAME --dataset $DATASET --seed $SEED --cuda $GPU_ID --patience $PATIENCE --data_root $DATA_ROOT

echo "Training completed."
