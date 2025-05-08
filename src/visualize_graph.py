import json
import networkx as nx
import matplotlib.pyplot as plt
import graph


def create_and_visualize_graph(data_or_path):
    """
    Constructs and visualizes a graph from JSON describing nodes and edges.

    Parameters:
    - data_or_path (str or dict): Path to the JSON file or a dict loaded from JSON.
    """
    # Load JSON data
    if isinstance(data_or_path, str):
        with open(data_or_path, 'r') as f:
            data = json.load(f)
    else:
        data = data_or_path

    # Build directed graph
    G = nx.DiGraph()

    # Add nodes with labels
    for node in data.get('nodes', []):
        node_id = node['id']
        label = ", ".join(node['labels'])
        G.add_node(node_id, label=label)

    # Add edges with labels
    for edge in data.get('edges', []):
        src = edge['source']
        dst = edge['target']
        elabel = ", ".join(edge['labels'])
        G.add_edge(src, dst, label=elabel)

    # Visualization
    pos = nx.spring_layout(G, seed=42)
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Graph Visualization from JSON")
    plt.axis('off')
    plt.show()

def visualize_graph(graph: graph, file: str = None):

    G = nx.DiGraph()

    for vertex in graph.vertices:
        G.add_node(vertex.uri, label=vertex.label)

    for edge in graph.edges:
        G.add_edge(edge.v1.uri, edge.v2.uri, label=edge.label)

    # Visualization
    pos = nx.spring_layout(G, seed=42)
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'label')

    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title("Graph Visualization from JSON")
    plt.axis('off')
    if file is None:
        plt.show()
    else:
        plt.savefig(file)


# create_and_visualize_graph('bob.json')
# Usage:
# 1. Save your canvas JSON content to a file, e.g., '/mnt/data/amy.json'
# 2. Call:
#    create_and_visualize_graph('/mnt/data/amy.json')
# Or, if you already have the JSON loaded into a dict named `amy_data`:
#    create_and_visualize_graph(amy_data)
