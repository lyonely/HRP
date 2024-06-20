import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLP(torch.nn.Module):
    def __init__(self, nfeature, nhid, nclass):
        super(MLP, self).__init__()
        self.conv1 = Linear(nfeature, nhid)
        self.conv2 = Linear(nhid, nclass)

    def forward(self, data):
        x = data.x
        x = F.relu(self.conv1(x))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x)
        return F.log_softmax(x, dim=1)

