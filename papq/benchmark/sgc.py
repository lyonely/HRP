import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class SGC(torch.nn.Module):
    def __init__(self, nfeature, nclass, K):
        super(SGC, self).__init__()
        self.conv1 = SGConv(nfeature, nclass, K=K, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

