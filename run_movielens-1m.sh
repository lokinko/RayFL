#!/bin/bash

pip install -r requirements.txt

# movielens-1m
python federated_train.py   --method pfedrec --dataset movielens-1m --data_file "1m-ratings.dat" --num_rounds 100 \
    --mu 0.01 --lambda 0.01 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.5 --lr_args 1e3 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedrap --dataset movielens-1m --data_file "1m-ratings.dat" --num_rounds 100 \
    --mu 0.01 --lambda 0.01 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.5 --lr_args 1e3 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpor --dataset movielens-1m --data_file "1m-ratings.dat" --num_rounds 100 \
    --mu 0.01 --lambda 0.01 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.5 --lr_args 1e3 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpora --dataset movielens-1m --data_file "1m-ratings.dat" --num_rounds 100 \
    --mu 0.01 --lambda 0.01 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.5 --lr_args 1e3 --num_workers 16
echo "Starting the program..."
sleep 1s