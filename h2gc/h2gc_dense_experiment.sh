#!/bin/bash

# Define an array of numbers
# numbers=(42 123 5 55 12345)
numbers=(111 222 333 444 555 666 777 888 999 1110)

# Loop through the array
for i in "${numbers[@]}"; do
    python h2gc_dense.py --dataset Flickr --K 2 --pre H2GC --seed $i --save_result --lr 0.001 --n_hidden 512
    python h2gc_dense.py --dataset Flickr --K 2 --pre SIGN --seed $i --save_result --lr 0.001 --n_hidden 512

    python h2gc_dense.py --dataset Reddit --K 2 --pre SIGN --seed $i --save_result --lr 0.00005 --n_hidden 512
    python h2gc_dense.py --dataset Reddit --K 2 --pre H2GC --seed $i --save_result --lr 0.00005 --n_hidden 512 --n_clusters 8
done