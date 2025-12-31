def decode(node_embeddings, edge_index):
    """
    Computes a score for each edge using
    dot product between node embeddings.
    """

    source_nodes, target_nodes = edge_index
    scores = (node_embeddings[source_nodes] *
              node_embeddings[target_nodes]).sum(dim=1)

    return scores
