from graph import Vertex, Graph
from visualize_graph import visualize_graph

def get_graph():
    v1: Vertex = Vertex('bob/v1', 'Bob')
    # v2: Vertex = Vertex('bob/v2', 'Bob')
    v3: Vertex = Vertex('bob/v3', 'Red')
    v4: Vertex = Vertex('bob/v4', 'Soccer')
    v5: Vertex = Vertex('bob/v5', 'Barcelona')
    v6: Vertex = Vertex('bob/v6', 'Spain')
    v7: Vertex = Vertex('bob/v7', 'Color')
    v8: Vertex = Vertex('bob/v8', 'FC Barcelona')
    v9: Vertex = Vertex('bob/v9', 'LaLiga')
    v10: Vertex = Vertex('bob/v10', 'Amy')

    user_profile: Graph = Graph()

    user_profile.add_vertex(v1)
    # user_profile.add_vertex(v2)
    user_profile.add_vertex(v3)
    user_profile.add_vertex(v4)
    user_profile.add_vertex(v5)
    user_profile.add_vertex(v6)
    user_profile.add_vertex(v7)
    user_profile.add_vertex(v8)
    user_profile.add_vertex(v9)
    user_profile.add_vertex(v10)

    # user_profile.add_edge(v1, v2, 'is named')

    user_profile.add_edge(v1, v3, 'likes')
    user_profile.add_edge(v3, v7, 'is a')

    user_profile.add_edge(v1, v5, 'born in')
    user_profile.add_edge(v5, v6, 'is in')

    user_profile.add_edge(v1, v8, 'follows')

    user_profile.add_edge(v1, v4, 'plays')

    user_profile.add_edge(v8, v5, 'play in')
    user_profile.add_edge(v8, v9, 'competes in')
    user_profile.add_edge(v8, v4, 'plays')

    user_profile.add_edge(v9, v6, 'is in')

    user_profile.add_edge(v1, v10, 'friends with')

    return user_profile

