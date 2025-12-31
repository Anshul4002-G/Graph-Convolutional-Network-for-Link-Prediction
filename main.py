import torch
import networkx as nx

from data.load_data import load_graph
from models.gcn import GCN
from training.train import train
from evaluation.evaluate import evaluate
from visualization.visualize import visualize_graph


def main():
    # Load graph and split edges
    train_data, val_data, test_data = load_graph()

    # Initialize model
    model = GCN(
        input_dim=train_data.num_node_features,
        hidden_dim=64
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train model
    for epoch in range(200):
        loss = train(model, train_data, optimizer, loss_fn)

        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Training Loss: {loss:.4f}")

    # Evaluate on test data
    precision, recall, auc = evaluate(model, test_data)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"AUC: {auc:.3f}")

    # Save metrics
    with open("results/metrics.txt", "w") as file:
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"AUC: {auc}\n")

    # Visualization
    graph = nx.karate_club_graph()
    predicted_edges = list(zip(
        test_data.edge_label_index[0].tolist(),
        test_data.edge_label_index[1].tolist()
    ))[:10]

    visualize_graph(
        graph,
        predicted_edges,
        "results/graph_visualization.html"
    )


if __name__ == "__main__":
    main()
