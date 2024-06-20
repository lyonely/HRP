#!/bin/bash

seeds=(111 222 333 444 555 666 777 888 999 1110)
glora_ranks=(0 1 2 4 8 16)

# Loop through the array

# Sparse experiments
# for i in "${seeds[@]}"; do
#     for j in "${glora_ranks[@]}"; do
#         python glora_sparse.py --dataset Cora --model GAT --glora_rank $j --seed $i --save_result
#         python glora_sparse.py --dataset Citeseer --model GAT --glora_rank $j --seed $i --save_result
#         python glora_sparse.py --dataset Pubmed --model GAT --glora_rank $j --seed $i --save_result

#         python glora_sparse.py --dataset Cora --model GCN --glora_rank $j --seed $i --save_result
#         python glora_sparse.py --dataset Citeseer --model GCN --glora_rank $j --seed $i --save_result
#         python glora_sparse.py --dataset Pubmed --model GCN --glora_rank $j --seed $i --save_result
#     done
# done

# Dense experiments
for i in "${seeds[@]}"; do
    for j in "${glora_ranks[@]}"; do
        python glora_dense.py --glora_rank $j --seed $i --save_result --epoch 30 --dataset Flickr
        python glora_dense.py --glora_rank $j --seed $i --save_result --epoch 30 --dataset Reddit
    done
done