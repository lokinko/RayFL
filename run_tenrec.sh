#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Default method (can be overridden via command line)
METHOD=${1:-"fedpor"}

# Optional comma-separated gamma values
GAMMA_VALUES=${2:-""}

# Common parameters for all runs
COMMON_PARAMS="--method $METHOD --dataset tenrec --data_file \"QB-article.csv\" --num_rounds 200 \
    --mu 0.1 --lambda 1e-4 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 1000 --num_workers 16"

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

# Usage: ./run_tenrec.sh [method] [gamma_values]
# Examples: 
# ./run_tenrec.sh pfedrec "0.01,0.3"
# ./run_tenrec.sh fedpor