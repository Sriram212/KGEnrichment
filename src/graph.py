from typing import List

class Entity:
    def __init__(self, uri, label):
        self.uri = uri
        self.label = label   

    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label

class Vertex(Entity):
    def __init__(self, uri, label):
        super().__init__(uri, label)
        self.outward_degree = 0
    def __eq__(self, other):
        return isinstance(other, Vertex) and self.uri == other.uri
    def __hash__(self):
        return hash(self.uri)


class Edge(Entity):
    def __init__(self, uri, v1: Vertex, v2: Vertex, label):
        super().__init__(uri, label)
        self.v1 = v1
        self.v2 = v2
    def __eq__(self, other):
        return (
            isinstance(other, Edge) and 
            self.uri == other.uri
        )

class Graph:
    def __init__(self):
        self.edges = []
        self.vertices = []

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    
    def remove_vertex(self, vertex):
        self.vertices.remove(vertex)
        self.edges = [edge for edge in self.edges if edge.v1 != vertex and edge.v2 != vertex]
    
    def add_edge(self, v1, v2, label=None):
        uri = f"{v1.uri}->{v2.uri}"
        edge = Edge(uri, v1, v2, label)
        v1.outward_degree += 1
        self.edges.append(edge)
        return edge
    
    def remove_edge(self, v1, v2):
        self.edges = [edge for edge in self.edges if not (edge.v1 == v1 and edge.v2 == v2)]
        v1.outward_degree -= 1

    def adjacent(self, v1, v2):
        return any(edge for edge in self.edges if edge.v1 == v1 and edge.v2 == v2)
    
    def neighbors(self, vertex):
        return {edge.v2 for edge in self.edges if edge.v1 == vertex}

    def lookup(self, uri):
        if '->' in uri:
            # Check for matching edge
            for edge in self.edges:
                if edge.uri == uri:
                    return edge
        else:
            # Check for matching vertex
            for vertex in self.vertices:
                if vertex.uri == uri:
                    return vertex
        # If not found
        return None

    def extract_lineage_set(self, v: Vertex):
        lineage_graph = Graph()
        visited = set()

        def dfs(current_vertex):
            if current_vertex in visited:
                return
            visited.add(current_vertex)

            # Add current vertex to lineage graph
            lineage_graph.add_vertex(current_vertex)

            # Get outward edges
            for edge in self.get_edges(current_vertex):
                next_vertex = edge.v2
                # lineage_graph.add_vertex(next_vertex)
                lineage_graph.add_edge(current_vertex, next_vertex, edge.label)
                dfs(next_vertex)

        dfs(v)
        return lineage_graph

    def get_edges(self, v: Vertex) -> List[Edge]:
        edge_list = []
        for edge in self.edges:
            if edge.uri.split("->")[0] == v.uri:
                edge_list.append(edge)

        return edge_list

    def print_graph(self):
        print("Graph:")
        for vertex in self.vertices:
            outgoing = [edge for edge in self.edges if edge.v1 == vertex]
            if outgoing:
                print(f"  {vertex.label} ({vertex.uri}) ->")
                for edge in outgoing:
                    print(f"    └── {edge.v2.label} ({edge.v2.uri}) [label: {edge.label}]")
            else:
                print(f"  {vertex.label} ({vertex.uri}) -> ∅")


