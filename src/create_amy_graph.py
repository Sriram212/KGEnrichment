import json
from graph import Vertex, Graph
from visualize_graph import visualize_graph

# Opening JSON file
def get_graph():
    with open('amy.json') as json_file:
        data = json.load(json_file)

    amy_graph = Graph()

    for node in data["nodes"]:
        uri = node["id"]
        label = node["labels"][0]
        amy_graph.add_vertex(Vertex(f"amy/{uri}", label))

    for edge in data["edges"]:
        src_uri = "amy/" + edge['source']
        tgt_uri = "amy/" + edge["target"]
        label = edge["labels"][0]
        src_vertex = amy_graph.lookup(src_uri)
        tgt_vertex = amy_graph.lookup(tgt_uri)
        amy_graph.add_edge(src_vertex, tgt_vertex, label)

    return amy_graph

