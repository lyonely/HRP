from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch
from tqdm import tqdm

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, classification_type='multiclass'):
        super(SAGE, self).__init__()

        self.num_layers = 2
        self.classification_type = classification_type
        self.convs0 = SAGEConv(in_channels, hidden_channels)
        self.convs1 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x_i):
        x, adjs = x_i
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = getattr(self, f"convs{i}")((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        if self.classification_type == 'multilabel':
            return x.sigmoid()

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
    
class PASAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, classification_type='multiclass'):
        super(PASAGE, self).__init__()

        self.classification_type = classification_type
        self.num_layers = 1
        self.convs1 = SAGEConv(in_channels, out_channels)

    def forward(self, x_i):
        x, adj = x_i
        edge_index, _, size = adj

        x_target = x[:size[1]] # Target nodes are always placed first.
        x = self.convs1((x, x_target), edge_index)

        if self.classification_type == 'multilabel':
            return x.sigmoid()
        return x.log_softmax(dim=-1)

    def inference(self, x_all, device=None, subgraph_loader=None, **kwargs):
        pbar = tqdm(total=x_all.size(0))
        pbar.set_description('Evaluating')

        x_all = x_all.to(device)

        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj.to(device)
            x = x_all[n_id].to(device)
            x_target = x[:size[1]]
            x = self.convs1((x, x_target), edge_index)
            xs.append(x.cpu())

            pbar.update(batch_size)

        x_all = torch.cat(xs, dim=0).to(device)

        pbar.close()

        return x_all