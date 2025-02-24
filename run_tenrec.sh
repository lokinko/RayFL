#!/bin/bash

pip install -r requirements.txt

# tenrec
python federated_train.py   --method pfedrec --dataset tenrec --data_file "QB-article.csv" --num_rounds 100 \
     --mu 0.1 --lambda 1e-4 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 1000 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedrap --dataset tenrec --data_file "QB-article.csv" --num_rounds 100 \
     --mu 0.1 --lambda 1e-4 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 1000 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpor --dataset tenrec --data_file "QB-article.csv" --num_rounds 100 \
     --mu 0.1 --lambda 1e-4 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 1000 --num_workers 16
echo "Starting the program..."
sleep 1s

python federated_train.py   --method fedpora --dataset tenrec --data_file "QB-article.csv" --num_rounds 100 \
     --mu 0.1 --lambda 1e-4 --gamma 0.01 --l2_regularization 1e-4 --lr_network 0.01 --lr_args 1000 --num_workers 16
echo "Starting the program..."
sleep 1s