"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 residual=False
    ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(p=0.5)
        self.classifier1 = nn.Linear(2 *num_hidden*8, in_dim, bias=False)
        # GCN layer (no residual)
        self.gat_layers.append(GraphConv(in_dim, num_hidden, activation=self.activation, norm='none'))
        # GAT layer (no residual)
        self.gat_layers.append(GATConv(
            num_hidden, num_classes, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))


    def forward(self, inputs, g):
        h = inputs
        h0 = self.gat_layers[0](g, h, edge_weights=g.edata['edge_weight'])
        h1 = self.gat_layers[1](g, h0).mean(1)
        return h1