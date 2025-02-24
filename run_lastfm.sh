#!/bin/bash

pip install -r requirements.txt

# last.fm
python federated_train.py   --method pfedrec --dataset last.fm --data_file "2k-ratings.dat" --num_rounds 100 \
    --mu 1.0 --lambda 0.1 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedrap --dataset last.fm --data_file "2k-ratings.dat" --num_rounds 100 \
    --mu 1.0 --lambda 0.1 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpor --dataset last.fm --data_file "2k-ratings.dat" --num_rounds 100 \
    --mu 1.0 --lambda 0.1 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpora --dataset last.fm --data_file "2k-ratings.dat" --num_rounds 100 \
    --mu 1.0 --lambda 0.1 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s