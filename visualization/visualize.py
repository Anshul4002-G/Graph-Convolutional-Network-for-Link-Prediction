import networkx as nx
import plotly.graph_objects as go


def visualize_graph(graph, predicted_edges, output_file):
    """
    Creates an interactive and visually clear graph showing:
    - Existing edges
    - Predicted links (dashed)
    """

    # Compute layout
    pos = nx.spring_layout(graph, seed=42)

    # ---------------- Existing Edges ----------------
    edge_x, edge_y = [], []
    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # ---------------- Predicted Edges ----------------
    pred_x, pred_y = [], []
    for u, v in predicted_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        pred_x += [x0, x1, None]
        pred_y += [y0, y1, None]

    # ---------------- Nodes ----------------
    node_x, node_y = [], []
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Create Plotly figure
    fig = go.Figure()

    # Existing edges
    fig.add_trace(go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="lightgray", width=2),
        name="Existing Links"
    ))

    # Predicted edges
    fig.add_trace(go.Scatter(
        x=pred_x,
        y=pred_y,
        mode="lines",
        line=dict(color="red", width=2, dash="dash"),
        name="Predicted Links"
    ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[f"Node {i}" for i in graph.nodes()],
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=14,
            color="royalblue",
            line=dict(width=2, color="black")
        ),
        name="Nodes"
    ))

    # Layout settings
    fig.update_layout(
        title="GCN Link Prediction on Social Network Graph",
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    # Save to HTML
    fig.write_html(output_file)
