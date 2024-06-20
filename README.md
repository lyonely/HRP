# HRP

Here we present the code for HRP, a novel three-stage framework for scalable training of GNNs. It consists of three different methods as follows.

1. **Homophily-Heterophily Graph Convolution** (H2GC), a GNN simplification preprocessing method designed to handle graphs with high heterophily.
2. **Graph Low Rank Approximation** (GLoRA), a technique that applies low-rank weight approximation to popular GNN models.
3. **Post Aggregation Product Quantization** (PAPQ), a data compression method to reduce storage requirements and speed up the training process.

The implementation of each of the methods can be found in their respective subfolders.

## Set-up

To run HRP, install the required libraries and modules with the following command.

`pip install -r requirements.txt`

## Acknowledgements
This project includes code from [LoRA](https://github.com/microsoft/LoRA), which is licensed under the [MIT License](https://github.com/microsoft/LoRA/blob/main/LICENSE.md), and code from [EPQuant](https://github.com/Lyun-Huang/EPQuant).

## Credit
Author: Yufeng Zhang
