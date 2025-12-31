import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from models.decoder import decode


@torch.no_grad()
def evaluate(model, data):
    """
    Evaluates the model on validation or test data.
    """

    model.eval()

    embeddings = model(data.x, data.edge_index)
    logits = decode(embeddings, data.edge_label_index)

    probabilities = torch.sigmoid(logits)
    predictions = (probabilities > 0.5).int()

    precision = precision_score(data.edge_label, predictions)
    recall = recall_score(data.edge_label, predictions)
    auc = roc_auc_score(data.edge_label, probabilities)

    return precision, recall, auc
