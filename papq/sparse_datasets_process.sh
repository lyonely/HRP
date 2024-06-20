#!/bin/bash

python epq_alone.py --dataset='Cora' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./pq_data/cora/ --reduce
python epq_alone.py --dataset='Pubmed' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./pq_data/pubmed/ --reduce
python epq_alone.py --dataset='Citeseer' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./pq_data/citeseer/ --reduce

python epq_alone.py --dataset='Cora' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/cora_sign/ --reduce --pre SIGN
python epq_alone.py --dataset='Pubmed' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/pubmed_sign/ --reduce --pre SIGN
python epq_alone.py --dataset='Citeseer' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/citeseer_sign/ --reduce --pre SIGN

python epq_alone.py --dataset='Cora' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/cora_sgc/ --reduce --pre SGC
python epq_alone.py --dataset='Pubmed' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/pubmed_sgc/ --reduce --pre SGC
python epq_alone.py --dataset='Citeseer' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/citeseer_sgc/ --reduce --pre SGC

python epq_alone.py --dataset='Cora' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/cora_h2gc/ --reduce --pre H2GC
python epq_alone.py --dataset='Pubmed' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/pubmed_h2gc/ --reduce --pre H2GC
python epq_alone.py --dataset='Citeseer' --try_cluster=30 --n_iter=30 --block_size 8 --ncents 128 --path ./papq_data/citeseer_h2gc/ --reduce --pre H2GC

# Root directory where the search should start
# Replace '/path/to/root/directory' with the path where your folders are located
root_dir="."

# Define old and new names in pairs
declare -a files_to_rename=(
    "blksize_8_batchsiz_1024_ncents_128_assignments.pth blksize_8_assignments.pth"
    "blksize_8_batchsiz_1024_ncents_128_centroids.pth blksize_8_centroids.pth"
)

# Loop through each pair and perform the renaming
for pair in "${files_to_rename[@]}"; do
    read -r old_name new_name <<< "$pair"  # Split the pair into old and new names

    # Find and rename files
    find "$root_dir" -type f -name "$old_name" -exec sh -c '
        for file do
            # Construct new file path by replacing the original file name
            new_file=$(dirname "$file")/'"$new_name"'

            # Rename the file
            mv "$file" "$new_file"
            echo "Renamed $file to $new_file"
        done
    ' sh {} +
done
