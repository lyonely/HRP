# PAPQ
This folder contains our implementation of **Post Aggregation Product Quantization** (PAPQ).

## Figure
![PAPQ Figure](https://github.com/lyonely/HRP/assets/66782598/dbbaa2f9-7a2e-47ca-87b1-7b1ecf5f39e5)

## File Structure
The file structure for this folder is as follows:
`epq_alone.py` - used for conducting product quantization on the datasets
`epq_sparse.py` - run the model on sparse datasets
`papq_dense.py` - run the model on dense datasets
`h2gc.py`, `uspec.py` - used for H2GC preprocessing
`sparse_datasets_process.sh`, `dense_datasets_process.sh` - script to process the datasets for experiments
`sparse_datasets_experiment.sh`, `dense_datasets_experiment.sh` - script to run the experiments
`/benchmark` - folder containing the model implementations
`/quantization` - folder containing code for product quantization

## Running Experiments
To run the experiments, users will need to run the code in the following order:
```
./sparse_datasets_process.sh
```
```
./dense_datasets_process.sh
```
to first generate the quantized datasets for experiments. Ensure you create the folders for the datasets first before running the scripts.
```
./sparse_datasets_experiment.sh
```
```
./dense_datasets_experiment.sh
```
to run the experiments on the quantized datasets.

To quantize the datasets individually, users can use the following command:
```
python epq_alone.py --dataset {dataset} --reduce {to_reduce} --seed {seed} --pr_thread {pr_thread} --block_size {block_size} --ncents {num_centroids} --try_cluster {try_cluster} --n_iter {n_iter} --batch_size {batch_size} --n_clusters {n_clusters} --path{path} --pre {pre} --K {K} --seed {seed}
```
with parameters as follows:
- {dataset}: dataset to process, options include Cora, Citeseer, Pubmed, Actor, Cornell, Texas, Chameleon, Squirrel, Flickr, Reddit
- {to_reduce}: boolean to indicate whether to do zero-vector exclusion for the dataset
- {pr_thread}: number of parallel runs
- {block_size}: block size for product quantization
- {num_centroids}: number of centroids to find in each clustering task
- {try_cluster}: number of attempts to find more centroids
- {n_iter}: number of iterations for the clustering algorithm
- {batch_size}: number of data points for each clustering task in each batch
- {n_clusters}: number of clusters to use for H2GC preprocessing
- {path}: path to save the processed dataset
- {pre}: type of preprocessing to use, options include H2GC, SIGN and SGC
- {K}: depth for the preprocessing method
- {seed}: integer indicating seed for the experiment


After processing the dataset, users can use the following commands to run the model:
```
python epq_sparse.py --dataset {dataset} --model {model_type} --lr {learning_rate} --wd {weight_decay} --hidden {hidden_channels} --epoch {epochs} --seed {seed} --save_result {save_result} --pqnt {pqnt} --path {path} --pre {pre} --quant_type {quant_type} --K {K}
```
with parameters as follows:
- {dataset}: dataset to train on, options include Cora, Citeseer, Pubmed, Actor, Cornell, Texas, Chameleon, Squirrel
- {model_type}: type of model to use, choice between GCN and MLP
- {learning_rate}: float indicating learning rate of the model
- {weight_decay}: float indicating weight decay of the optimizer
- {hidden_channels}: integer indicating number of channels in the hidden layer
- {epochs}: integer indicating number of epochs to train for
- {seed}: integer indicating seed for the experiment
- {save_result}: boolean indicating whether to save the results in a csv file
- {pqnt}: boolean indicating whether to use the quantized data
- {path}: path to the quantized data
- {quant_type}: type of quantization used, choice between EPQ and PAPQ
- {pre}: type of preprocessing to use, options include H2GC, SIGN and SGC
- {K}: depth for the preprocessing method

```
python papq_dense.py --datset {dataset} --lr {learning_rate} --wd {weight_decay} --n_hidden {hidden_channels} --epoch {epochs} --batch_size {batch_size} --seed {seed} --save_result {save_result} --pqnt {pqnt} --path {path} --pre {pre} --quant_type {quant_type} --K {K}
```
with parameters as follows:
- {dataset}: dataset to train on, options include Flickr, Reddit
- {learning_rate}: float indicating learning rate of the model
- {weight_decay}: float indicating weight decay of the optimizer
- {hidden_channels}: integer indicating number of channels in the hidden layer
- {epochs}: integer indicating number of epochs to train for
- {batch_size}: integer indicating size of batch for mini-batch training
- {seed}: integer indicating seed for the experiment
- {save_result}: boolean indicating whether to save the results in a csv file
- {pqnt}: boolean indicating whether to use the quantized data
- {path}: path to the quantized data
- {quant_type}: type of quantization used, choice between EPQ and PAPQ
- {pre}: type of preprocessing to use, options include H2GC, SIGN and SGC
- {K}: depth for the preprocessing method
