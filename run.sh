#!/bin/bash

pip install -r requirements.txt

python federated_train.py --method $1 --dataset $2 --num_rounds $3 --optimizer $4 --num_workers 16 --seed 0