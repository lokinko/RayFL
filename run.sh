#!/bin/bash

# Master script to run experiments across all datasets

# Install required packages
pip install -r requirements.txt

# Default values for command-line arguments
DATASET=${1:-"all"}       # Options: movielens-1m, amazon, tenrec, last.fm, all
METHOD=${2:-"fedpor"}     # Default method
GAMMA=${3:-""}            # Optional comma-separated gamma values

# Optional override parameters
MU=${4:-""}               # Optional mu value
LAMBDA=${5:-""}           # Optional lambda value
L2_REG=${6:-""}           # Optional l2_regularization value
LR_NET=${7:-""}           # Optional lr_network value
LR_ARGS=${8:-""}          # Optional lr_args value
NUM_ROUNDS=${9:-"100"}    # Optional num_rounds value
NUM_WORKERS=${10:-"16"}   # Number of workers (default: 16)

# Function to run experiments for a specific dataset
run_experiment() {
    local dataset=$1
    local method=$2
    local gamma=$3
    
    echo "==================================================="
    echo "Running experiments for dataset: $dataset"
    echo "==================================================="
    
    # Dataset-specific default parameters
    local default_mu default_lambda default_l2_reg default_lr_net default_lr_args
    
    case $dataset in
        "movielens-1m")
            DATA_FILE="1m-ratings.dat"
            default_mu="0.01"
            default_lambda="0.01"
            default_l2_reg="1e-4"
            default_lr_net="0.5"
            default_lr_args="1e3"
            ;;
        "amazon")
            DATA_FILE="video-ratings.csv"
            default_mu="0.01"
            default_lambda="0.001"
            default_l2_reg="1e-3"
            default_lr_net="0.01"
            default_lr_args="100"
            ;;
        "tenrec")
            DATA_FILE="QB-article.csv"
            default_mu="0.1"
            default_lambda="1e-4"
            default_l2_reg="1e-4"
            default_lr_net="0.01"
            default_lr_args="1000"
            ;;
        "last.fm")
            DATA_FILE="2k-ratings.dat"
            default_mu="1.0"
            default_lambda="0.1"
            default_l2_reg="1e-4"
            default_lr_net="0.01"
            default_lr_args="100"
            ;;
        *)
            echo "Unknown dataset: $dataset"
            return 1
            ;;
    esac
    
    # Use provided values or defaults
    local mu_param=${MU:-$default_mu}
    local lambda_param=${LAMBDA:-$default_lambda}
    local l2_reg_param=${L2_REG:-$default_l2_reg}
    local lr_net_param=${LR_NET:-$default_lr_net}
    local lr_args_param=${LR_ARGS:-$default_lr_args}
    
    # Build full parameter string - fixed parameter handling
    COMMON_PARAMS="--dataset $dataset --data_file $DATA_FILE --num_rounds $NUM_ROUNDS "
    COMMON_PARAMS+="--mu $mu_param --lambda $lambda_param --l2_regularization $l2_reg_param "
    COMMON_PARAMS+="--lr_network $lr_net_param --lr_args $lr_args_param"
    
    FULL_PARAMS="--method $method $COMMON_PARAMS --num_workers $NUM_WORKERS"
    
    echo "Using parameters:"
    echo "  method: $method"
    echo "  dataset: $dataset"
    echo "  data_file: $DATA_FILE"
    echo "  mu: $mu_param"
    echo "  lambda: $lambda_param"
    echo "  l2_regularization: $l2_reg_param"
    echo "  lr_network: $lr_net_param"
    echo "  lr_args: $lr_args_param"
    echo "  num_rounds: $NUM_ROUNDS"
    echo "  num_workers: $NUM_WORKERS"
    
    # Debug: show actual command to be executed
    echo "Command to execute:"
    
    # Check if gamma values were provided
    if [ -z "$gamma" ]; then
        # Run once without specifying gamma
        echo "python federated_train.py $FULL_PARAMS"
        python federated_train.py $FULL_PARAMS
    else
        # Run with each provided gamma value
        IFS=',' read -ra GAMMA_ARRAY <<< "$gamma"
        for G in "${GAMMA_ARRAY[@]}"; do
            echo "python federated_train.py $FULL_PARAMS --gamma $G"
            python federated_train.py $FULL_PARAMS --gamma $G
            echo "Program completed..."
            sleep 1s
        done
    fi
}

# Run experiments based on selected dataset
if [ "$DATASET" == "all" ]; then
    # Run experiments for all datasets
    for ds in "movielens-1m" "amazon" "tenrec" "last.fm"; do
        run_experiment "$ds" "$METHOD" "$GAMMA"
    done
else
    # Run experiments for the specified dataset
    run_experiment "$DATASET" "$METHOD" "$GAMMA"
fi

echo "All experiments completed!"

# Usage instructions
cat << EOF

Usage: ./run.sh [dataset] [method] [gamma] [mu] [lambda] [l2_reg] [lr_net] [lr_args] [num_rounds] [num_workers]

Parameters:
  dataset      - The dataset to use (movielens-1m, amazon, tenrec, last.fm, or all)
  method       - The federated learning method (fedpor, fedpora, fedrap, pfedrec, etc.)
  gamma        - Optional comma-separated gamma values (e.g. "0.01,0.1,1.0")
  mu           - Optional mu parameter value (defaults to dataset-specific value)
  lambda       - Optional lambda parameter value (defaults to dataset-specific value)
  l2_reg       - Optional l2_regularization parameter value (defaults to dataset-specific value)
  lr_net       - Optional lr_network parameter value (defaults to dataset-specific value)
  lr_args      - Optional lr_args parameter value (defaults to dataset-specific value)
  num_rounds   - Optional number of rounds (default: 100)
  num_workers  - Optional number of workers (default: 16)

Examples:
  ./run.sh movielens-1m fedpor "0.01,1.0"    # Run fedpor on movielens-1m with gamma values 0.01 and 1.0
  ./run.sh amazon fedpor "" 0.1              # Run fedpor on amazon with mu=0.1 (other params at default)
  ./run.sh all fedpora "0.01" "" 0.2 "" 0.05  # Run fedpora on all datasets with gamma=0.01, lambda=0.2, lr_network=0.05

EOF
