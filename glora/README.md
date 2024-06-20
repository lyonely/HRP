# GLoRA
This folder contains our implementation of **Graph Low-Rank Approximation** (GLoRA).

## Figure
![GLoRA Figure](https://github.com/lyonely/HRP/assets/66782598/6f0ad3f1-caa9-4a43-a352-0fc231fd7499)

## File Structure
The main directory contains the code for running the experiments and models.
- `glora_experiment.sh` - runs the whole experiment
- `glora_sparse.sh` - runs the model for sparse datasets
- `glora_dense.sh` - runs the model for dense datasets

The subdirectory, `/glora_benchmark`, contains the implementation for the GLoRA-variant GNN models.
- `gat.py` - our GLoRA implementation of Graph Attention Network
- `gcn.py` - our GLoRA implementation of Graph Convolutional Network
- `sage.py` - our GLoRA implementation of GraphSAGE

## Running Experiments
To run the experiments, users can run the following command:
```
./glora_experiment.sh
```

To run the model individually, users can use the following commands:
```
python glora_sparse.py --datset {dataset} --model {model_type} --lr {learning_rate} --wd {weight_decay} --n_hidden {hidden_channels} --epoch {epochs} --nheads {num_heads} --glora_rank {rank} --seed {seed} --save_result {save_result}
```
with parameters as follows:
- {dataset}: dataset to train on, options include Cora, Citeseer, Pubmed, Actor, Cornell, Texas, Chameleon, Squirrel
- {learning_rate}: float indicating learning rate of the model
- {model_type}: model to use, choice between GCN and GAT
- {weight_decay}: float indicating weight decay of the optimizer
- {hidden_channels}: integer indicating number of channels in the hidden layer
- {epochs}: integer indicating number of epochs to train for
- {num_heads}: integer indicating number of attention heads for GAT
- {rank}: integer indicating rank for weight decomposition in GLoRA
- {seed}: integer indicating seed for the experiment
- {save_result}: boolean indicating whether to save the results in a csv file

```
python glora_dense.py --datset {dataset} --lr {learning_rate} --wd {weight_decay} --n_hidden {hidden_channels} --epoch {epochs} --batch_size {batch_size} --glora_rank {rank} --seed {seed} --save_result {save_result}
```
with parameters as follows:
- {dataset}: dataset to train on, options include Cora, Citeseer, Pubmed, Actor, Cornell, Texas, Chameleon, Squirrel
- {learning_rate}: float indicating learning rate of the model
- {weight_decay}: float indicating weight decay of the optimizer
- {hidden_channels}: integer indicating number of channels in the hidden layer
- {epochs}: integer indicating number of epochs to train for
- {batch_size}: integer indicating size of batch for mini-batch training
- {rank}: integer indicating rank for weight decomposition in GLoRA
- {seed}: integer indicating seed for the experiment
- {save_result}: boolean indicating whether to save the results in a csv file



