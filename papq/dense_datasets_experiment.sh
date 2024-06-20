#!/bin/bash

# Define an array of numbers
numbers=(111 222 333 444 555 666 777 888 999 1110)

# Loop through the array
for i in "${numbers[@]}"; do

    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --save_result --seed $i --pre SGC
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --save_result --seed $i --pre SIGN
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --save_result --seed $i --pre H2GC
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --pqnt --save_result --seed $i --path ./pq_data/reddit/ --pre SGC --quant_type EPQ
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --pqnt --save_result --seed $i --path ./pq_data/reddit/ --pre SIGN --quant_type EPQ
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --pqnt --save_result --seed $i --path ./pq_data/reddit/ --pre H2GC --quant_type EPQ
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --pqnt --save_result --seed $i --path ./papq_data/reddit_sgc/ --pre SGC --quant_type PAPQ
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --pqnt --save_result --seed $i --path ./papq_data/reddit_sign/ --pre SIGN --quant_type PAPQ
    python papq_dense.py --block_size 32 --ncents 512 --dataset='Reddit' --pqnt --save_result --seed $i --path ./papq_data/reddit_h2gc/ --pre H2GC --quant_type PAPQ

    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --save_result --seed $i --pre SGC
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --save_result --seed $i --pre SIGN
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --save_result --seed $i --pre H2GC
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --pqnt --save_result --seed $i --path ./pq_data/flickr/ --pre SGC --quant_type EPQ
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --pqnt --save_result --seed $i --path ./pq_data/flickr/ --pre SIGN --quant_type EPQ
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --pqnt --save_result --seed $i --path ./pq_data/flickr/ --pre H2GC --quant_type EPQ
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --pqnt --save_result --seed $i --path ./papq_data/flickr_sgc/ --pre SGC --quant_type PAPQ
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --pqnt --save_result --seed $i --path ./papq_data/flickr_sign/ --pre SIGN --quant_type PAPQ
    python papq_dense.py --block_size 16 --ncents 256 --dataset='Flickr' --pqnt --save_result --seed $i --path ./papq_data/flickr_h2gc/ --pre H2GC --quant_type PAPQ
    
done