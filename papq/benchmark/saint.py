import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class SAINT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, classification_type='multiclass'):
        super().__init__()
        self.classification_type = classification_type
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index):
        x1 = F.relu(self.conv1(x0, edge_index))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        if self.classification_type == 'multilabel':
            return x.sigmoid()
        return x.log_softmax(dim=-1)
    
    def inference(self, x_all, edge_index=None, **kwargs):
        return self.forward(x_all, edge_index)