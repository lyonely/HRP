# H2GC
This folder contains our implementation of **Homophily-Heterophily Graph Convolution** (H2GC).

## Figure
![H2GC Diagram](https://github.com/lyonely/HRP/assets/66782598/38d6c6c9-0571-48a3-80cc-27d0de75b10d)

## Executing Experiments

To run the experiments, users can run the following commands:
```
./h2gc_sparse_experiment.sh
```
```
./h2gc_dense_experiment.sh
```
which will run the experiments for the sparse and dense datasets respectively.

To run the model individually, users can use the following command:

```
python h2gc_sparse.py --dataset {dataset} --lr {learning_rate} --wd {weight_decay} --n_hidden {hidden_channels} --epoch {epochs} --K {depth} --pre {preprocessing method} --n_clusters {clusters} --seed {seed} --save_result {save_result}
```
with parameters as follows:
- {dataset}: dataset to train on, options include Cora, Citeseer, Pubmed, Actor, Texas, Cornell, Chameleon and Squirrel
- {learning_rate}: float indicating learning rate of the model
- {weight_decay}: float indicating weight decay of the optimizer
- {hidden_channels}: integer indicating number of channels in the hidden layer
- {epochs}: integer indicating number of epochs to train for
- {depth}: integer indicating number of hops for preprocessing
- {preprocessing method}: choice between SGC, SIGN and H2GC for preprocessing the data
- {clusters}: integer indiciating number of clusters for the clustering algorithm in H2GC
- {seed}: integer indicating seed for the experiment
- {save_result}: boolean indicating whether to save the results in a csv file

```
python h2gc_dense.py --dataset {dataset} --lr {learning_rate} --wd {weight_decay} --n_hidden {hidden_channels} --epoch {epochs} --batch_size {batch_size} --K {depth} --pre {preprocessing method} --n_clusters {clusters} --seed {seed} --save_result {save_result}
```
with parameters as follows:
- {dataset}: dataset to train on, options include Reddit and Flickr
- {learning_rate}: float indicating learning rate of the model
- {weight_decay}: float indicating weight decay of the optimizer
- {hidden_channels}: integer indicating number of channels in the hidden layer
- {epochs}: integer indicating number of epochs to train for
- {batch_size}: integer indicating size of the mini-batch during mini-batch training
- {depth}: integer indicating number of hops for preprocessing, choice between 1 to 2
- {preprocessing method}: choice between SGC, SIGN and H2GC for preprocessing the data
- {clusters}: integer indiciating number of clusters for the clustering algorithm in H2GC
- {seed}: integer indicating seed for the experiment
- {save_result}: boolean indicating whether to save the results in a csv file

