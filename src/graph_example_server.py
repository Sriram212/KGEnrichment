from graph import Vertex, Graph
from visualize_graph import visualize_graph


def get_graph():
    v1: Vertex = Vertex('amy/v1', 'Amy')
    # v2: Vertex = Vertex('amy/v2', 'Amy')
    v3: Vertex = Vertex('amy/v3', 'Blue')
    v4: Vertex = Vertex('amy/v4', 'Soccer')
    v5: Vertex = Vertex('amy/v5', 'Bob')
    # v6: Vertex = Vertex('amy/v6', 'Red')
    # v7: Vertex = Vertex('amy/v7', 'Color')

    user_profile: Graph = Graph()

    user_profile.add_vertex(v1)
    # user_profile.add_vertex(v2)
    user_profile.add_vertex(v3)
    user_profile.add_vertex(v4)
    user_profile.add_vertex(v5)
    # user_profile.add_vertex(v6)
    # user_profile.add_vertex(v7)

    # user_profile.add_edge(v1, v2, 'is named')

    user_profile.add_edge(v1, v3, 'likes')
    # user_profile.add_edge(v3, v7, 'is a')

    user_profile.add_edge(v1, v4, 'plays')

    user_profile.add_edge(v1, v5, 'friends with')

    # user_profile.add_edge(v5, v6, 'likes')
    # user_profile.add_edge(v5, v4, 'plays')

    # user_profile.add_edge(v6, v7, 'is a')

    return user_profile

