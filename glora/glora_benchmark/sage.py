import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .lora import LoraLinear
from torch_geometric import EdgeIndex
from torch_scatter import scatter_add
from torch_geometric.utils._scatter import scatter

from tqdm import tqdm

class GraphSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank=0):
        super(GraphSAGEConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        linear_layer = torch.nn.Linear if rank == 0 else partial(LoraLinear, r=rank, lora_dropout=0.5)
        self.lin_l = linear_layer(in_channels, out_channels)
        self.lin_r = linear_layer(in_channels, out_channels, bias=False)

    def forward(self, x, edge_index):

        # Perform message passing
        x, x_target = x

        row, col = edge_index

        aggr_out = scatter(x[row], col, dim_size=x_target.size(0),reduce='mean')
        out = self.lin_l(aggr_out) + self.lin_r(x_target)

        return out
    
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rank=0):
        super(SAGE, self).__init__()

        self.num_layers = 2
        self.convs0 = GraphSAGEConv(in_channels, hidden_channels, rank=rank)
        self.convs1 = GraphSAGEConv(hidden_channels, out_channels, rank=0)

    def forward(self, x_i):
        x, adjs = x_i
        for i, (edge_index, _, size) in enumerate(adjs):
            edge_index = EdgeIndex(edge_index, sort_order='row')
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = getattr(self, f"convs{i}")((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader=None, device=None, **kwargs):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = getattr(self, f"convs{i}")((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all
