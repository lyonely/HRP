import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from functools import partial
from .lora import LoraLinear

class GATLayer(nn.Module):

    src_nodes_dim = 0
    dst_nodes_dim = 1

    nodes_dim = 0
    head_dim = 1

    def __init__(self, in_features, out_features, num_heads, dropout, concat=True, rank=0):
        super(GATLayer, self).__init__()
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat

        linear_layer = torch.nn.Linear if rank == 0 else partial(LoraLinear, r=rank, lora_dropout=dropout)
        
        # Define linear layers for each head
        self.lin = linear_layer(in_features, out_features * num_heads, bias=False)

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, num_heads, out_features))
        self.att_dst = Parameter(torch.empty(1, num_heads, out_features))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        # Step 1: Linear Projection + Regularization
        assert adj.shape[0] == 2, f'Expected edge index with shape=(2,E) got {adj.shape}'
        in_nodes_features = self.dropout(x)

        nodes_features_proj = self.lin(in_nodes_features).view(-1, self.num_heads, self.out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        # Step 2: Compute attention coefficients
        scores_src = (nodes_features_proj * self.att_src).sum(dim=-1)
        scores_dst = (nodes_features_proj * self.att_dst).sum(dim=-1)

        scores_src_lifted, scores_dst_lifted, nodes_features_proj_lifted = self.lift(scores_src, scores_dst, nodes_features_proj, adj)

        scores_per_edge = F.leaky_relu(scores_src_lifted + scores_dst_lifted, negative_slope=0.2)

        attentions_per_edge = self.neighbourhood_aware_softmax(scores_per_edge, adj[self.dst_nodes_dim], x.size(0))
        attentions_per_edge = self.dropout(attentions_per_edge)

        # Step 3: Neighborhood Aggregation
        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, adj, in_nodes_features, x.size(0))

        if self.concat:
            out_nodes_features = out_nodes_features.contiguous().view(-1, self.num_heads * self.out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=1)

        return out_nodes_features
    
    def lift(self, scores_src, scores_dst, nodes_features_proj, adj):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = adj[self.src_nodes_dim]
        trg_nodes_index = adj[self.dst_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_src = scores_src.index_select(self.nodes_dim, src_nodes_index)
        scores_dst = scores_dst.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_src, scores_dst, nodes_features_matrix_proj_lifted
    
    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.dst_nodes_dim], nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features
    
    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        
        return this.expand_as(other)

    def sum_edge_scores_neighbourhood_aware(self, exp_scores_per_edge, dst_index, num_of_nodes):
        dst_index_broadcasted = self.explicit_broadcast(dst_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighbourhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighbourhood_sums.scatter_add_(self.nodes_dim, dst_index_broadcasted, exp_scores_per_edge)

        return neighbourhood_sums.index_select(self.nodes_dim, dst_index)

    def neighbourhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()

        neighbourhood_aware_denominator = self.sum_edge_scores_neighbourhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        attentions_per_edge = exp_scores_per_edge / (neighbourhood_aware_denominator + 1e-16)

        return attentions_per_edge.unsqueeze(-1)


class GAT(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, num_layers, num_heads=4, dropout=0.5, rank=0):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Define the GAT layers
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(n_features, n_hidden, num_heads, dropout, concat=True, rank=rank))
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(n_hidden * num_heads, n_hidden, num_heads, dropout, concat=True, rank=rank))
        self.layers.append(GATLayer(n_hidden * num_heads, n_classes, 1, dropout, concat=False, rank=0))

    def forward(self, x, adj):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, adj))
        x = self.layers[-1](x, adj)
        return F.log_softmax(x, dim=1)
