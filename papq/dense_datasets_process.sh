#!/bin/bash

python epq_alone.py --dataset="Reddit" --block_size 32 --try_cluster=30 --n_iter=15 --ncents 512 --path ./pq_data/reddit/
python epq_alone.py --dataset="Reddit" --block_size 32 --try_cluster=30 --n_iter=15 --ncents 512 --path ./papq_data/reddit_sgc/ --pre SGC
python epq_alone.py --dataset="Reddit" --block_size 32 --try_cluster=30 --n_iter=15 --ncents 512 --path ./papq_data/reddit_sign/ --pre SIGN --pr_thread 1

python epq_alone.py --dataset="Flickr" --block_size 16 --try_cluster=30 --n_iter=15 --ncents 256 --path ./pq_data/flickr/
python epq_alone.py --dataset="Flickr" --block_size 16 --try_cluster=30 --n_iter=15 --ncents 256 --path ./papq_data/flickr_sgc/ --pre SGC
python epq_alone.py --dataset="Flickr" --block_size 16 --try_cluster=30 --n_iter=15 --ncents 256 --path ./papq_data/flickr_sign/ --pre SIGN --pr_thread 1

python epq_alone.py --dataset="Reddit" --block_size 32 --try_cluster=30 --n_iter=15 --ncents 512 --path ./papq_data/reddit_h2gc/ --pre H2GC --pr_thread 1 --n_clusters 16
python epq_alone.py --dataset="Flickr" --block_size 16 --try_cluster=30 --n_iter=15 --ncents 256 --path ./papq_data/flickr_h2gc/ --pre H2GC --pr_thread 1 --n_clusters 10

# Root directory where the search should start
# Replace '/path/to/root/directory' with the path where your folders are located
root_dir="."

# Define old and new names in pairs
declare -a files_to_rename=(
    "blksize_32_batchsiz_1024_ncents_512_assignments.pth blksize_32_assignments.pth"
    "blksize_32_batchsiz_1024_ncents_512_centroids.pth blksize_32_centroids.pth"
    "blksize_16_batchsiz_1024_ncents_256_assignments.pth blksize_16_assignments.pth"
    "blksize_16_batchsiz_1024_ncents_256_centroids.pth blksize_16_centroids.pth"
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