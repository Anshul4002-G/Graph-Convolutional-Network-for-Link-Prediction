import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import RandomLinkSplit



def load_graph():
    """
    Loads the Karate Club graph and prepares it for
    link prediction using PyTorch Geometric.
    """

    # Load classic social network dataset
    graph = nx.karate_club_graph()

    # Convert NetworkX graph to PyG format
    data = from_networkx(graph)

    # Use identity matrix as node features
    # (each node has a unique feature vector)
    data.x = torch.eye(data.num_nodes)

    # Split edges into train / validation / test
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=True
    )

    return splitter(data)
