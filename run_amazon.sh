#!/bin/bash

pip install -r requirements.txt

# amazon
python federated_train.py   --method pfedrec --dataset amazon --data_file "video-ratings.csv" \
    --mu 0.01 --lambda 0.001 --gamma 1.0 --l2_regularization 1e-3 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedrap --dataset amazon --data_file "video-ratings.csv" \
    --mu 0.01 --lambda 0.001  --gamma 1.0 --l2_regularization 1e-3 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpor --dataset amazon --data_file "video-ratings.csv" \
    --mu 0.01 --lambda 0.001  --gamma 1.0 --l2_regularization 1e-3 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpora --dataset amazon --data_file "video-ratings.csv" \
    --mu 0.01 --lambda 0.001 --gamma 1.0 --l2_regularization 1e-3 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s