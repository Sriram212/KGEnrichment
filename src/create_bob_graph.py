import json
from graph import Vertex, Graph
from visualize_graph import visualize_graph

# Opening JSON file
def get_graph():
    with open('bob.json') as json_file:
        data = json.load(json_file)

    bob_graph = Graph()

    for node in data["nodes"]:
        uri = node["id"]
        label = node["labels"][0]
        bob_graph.add_vertex(Vertex(f"bob/{uri}", label))

    for edge in data["edges"]:
        src_uri = "bob/" + edge['source']
        tgt_uri = "bob/" + edge["target"]
        label = edge["labels"][0]
        src_vertex = bob_graph.lookup(src_uri)
        tgt_vertex = bob_graph.lookup(tgt_uri)
        bob_graph.add_edge(src_vertex, tgt_vertex, label)

    return bob_graph

