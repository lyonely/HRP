#!/bin/bash

# Define an array of numbers
# numbers=(42 123 5 55 12345)
numbers=(111 222 333 444 555 666 777 888 999 1110)
ks=(1 2 3 4)
pres=(SIGN HHSIGN)
datasets=(Cora Citeseer Pubmed Squirrel Chameleon Actor Texas Cornell Wisconsin)
# Loop through the array
for i in "${numbers[@]}"; do
    for k in "${ks[@]}"; do
        for pre in "${pres[@]}"; do
            for dataset in "${datasets[@]}"; do
                python h2gc_sparse.py --dataset $dataset --K $k --pre $pre --seed $i --save_result
            done
        done
    done
done