#!/bin/bash

pip install -r requirements.txt

# lastfm
python federated_train.py   --method pfedrec --dataset lastfm --data_file "QB-article.csv" \
    --mu 1.0 --lambda 0.1 --gamma 1.0 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedrap --dataset lastfm --data_file "QB-article.csv" \
    --mu 1.0 --lambda 0.1 --gamma 1.0 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpor --dataset lastfm --data_file "QB-article.csv" \
    --mu 1.0 --lambda 0.1 --gamma 1.0 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpora --dataset lastfm --data_file "QB-article.csv" \
    --mu 1.0 --lambda 0.1 --gamma 1.0 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 100 --num_workers 16
echo "Starting the program..."
sleep 1s