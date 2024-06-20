#!/bin/bash

# Define an array of numbers
numbers=(111 222 333 444 555 666 777 888 999 1110)

# Loop through the array
for i in "${numbers[@]}"; do

    python epq_sparse.py --block_size=8 --model='GCN' --dataset='Cora' --pqnt --seed $i --save_result --path ./pq_data/cora/ --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='GCN' --dataset='Pubmed' --pqnt --seed $i --save_result --path ./pq_data/pubmed/ --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='GCN' --dataset='Citeseer' --pqnt --seed $i --save_result --path ./pq_data/citeseer/ --quant_type EPQ

    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Cora' --pqnt --seed $i --save_result --path ./pq_data/cora/ --pre SIGN --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Pubmed' --pqnt --seed $i --save_result --path ./pq_data/pubmed/ --pre SIGN --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Citeseer' --pqnt --seed $i --save_result --path ./pq_data/citeseer/ --pre SIGN --quant_type EPQ

    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Cora' --pqnt --seed $i --save_result --path ./pq_data/cora/ --pre H2GC --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Pubmed' --pqnt --seed $i --save_result --path ./pq_data/pubmed/ --pre H2GC --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Citeseer' --pqnt --seed $i --save_result --path ./pq_data/citeseer/ --pre H2GC --quant_type EPQ

    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Cora' --seed $i --save_result --pre SGC
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Pubmed' --seed $i --save_result --pre SGC
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Citeseer' --seed $i --save_result --pre SGC

    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Cora' --pqnt --seed $i --save_result --path ./pq_data/cora/ --pre SGC --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Pubmed' --pqnt --seed $i --save_result --path ./pq_data/pubmed/ --pre SGC --quant_type EPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Citeseer' --pqnt --seed $i --save_result --path ./pq_data/citeseer/ --pre SGC --quant_type EPQ

    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Cora' --pqnt --seed $i --save_result --pre SIGN --path ./papq_data/cora_sign/ --quant_type PAPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Pubmed' --pqnt --seed $i --save_result --pre SIGN --path ./papq_data/pubmed_sign/ --quant_type PAPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Citeseer' --pqnt --seed $i --save_result --pre SIGN --path ./papq_data/citeseer_sign/ --quant_type PAPQ

    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Cora' --pqnt --seed $i --save_result --pre SGC --path ./papq_data/cora_sgc/ --quant_type PAPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Pubmed' --pqnt --seed $i --save_result --pre SGC --path ./papq_data/pubmed_sgc/ --quant_type PAPQ
    python epq_sparse.py --block_size=8 --model='MLP' --dataset='Citeseer' --pqnt --seed $i --save_result --pre SGC --path ./papq_data/citeseer_sgc/ --quant_type PAPQ

done