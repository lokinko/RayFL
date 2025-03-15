#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Default method (can be overridden via command line)
METHOD=${1:-"fedpor"}

# Optional comma-separated gamma values
GAMMA_VALUES=${2:-""}

# Common parameters for all runs
COMMON_PARAMS="--method $METHOD --dataset amazon --data_file \"video-ratings.csv\" --num_rounds 100 \
    --mu 0.01 --lambda 0.001 --l2_regularization 1e-3 --lr_network 0.01 --lr_args 100 --num_workers 16"

# Check if gamma values were provided
if [ -z "$GAMMA_VALUES" ]; then
    # Run once without specifying gamma
    echo "Running training with method = $METHOD, no gamma specified"
    python federated_train.py $COMMON_PARAMS
else
    # Run with each provided gamma value
    IFS=',' read -ra GAMMA_ARRAY <<< "$GAMMA_VALUES"
    for GAMMA in "${GAMMA_ARRAY[@]}"; do
        echo "Running training with method = $METHOD, gamma = $GAMMA"
        python federated_train.py $COMMON_PARAMS --gamma $GAMMA
        echo "Program started..."
        sleep 1s
    done
fi

# Usage: ./run_amazon.sh [method] [gamma_values]
# Examples: 
# ./run_amazon.sh fedavg "0.05,0.02"
# ./run_amazon.sh fedpor
