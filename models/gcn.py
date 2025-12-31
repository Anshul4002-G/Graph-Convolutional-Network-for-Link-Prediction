import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Simple 2-layer Graph Convolutional Network.
    It takes node features and graph structure
    and outputs node embeddings.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # First GCN layer
        x = self.gcn1(x, edge_index)
        x = F.relu(x)

        # Second GCN layer
        x = self.gcn2(x, edge_index)

        return x
