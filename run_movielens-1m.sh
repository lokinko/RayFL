#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Default method (can be overridden via command line)
METHOD=${1:-"fedpor"}

# Optional comma-separated gamma values
GAMMA_VALUES=${2:-""}

# Common parameters for all runs
COMMON_PARAMS="--method $METHOD --dataset movielens-1m --data_file \"1m-ratings.dat\" --num_rounds 100 \
    --mu 0.01 --lambda 0.01 --l2_regularization 1e-4 --lr_network 0.5 --lr_args 1e3 --num_workers 16"

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

# Usage: ./run_movielens-1m.sh [method] [gamma_values]
# Examples: 
# ./run_movielens-1m.sh fedavg "0.01,1.0"
# ./run_movielens-1m.sh fedrap