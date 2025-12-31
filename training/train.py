from models.decoder import decode


def train(model, data, optimizer, loss_fn):
    """
    Runs one training step.
    """

    model.train()
    optimizer.zero_grad()

    # Get node embeddings from GCN
    embeddings = model(data.x, data.edge_index)

    # Predict links
    predictions = decode(embeddings, data.edge_label_index)

    # Compute loss
    loss = loss_fn(predictions, data.edge_label.float())

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()
