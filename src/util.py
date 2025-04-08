import random
from tenseal import CKKSVector
from typing import List
from graph import Vertex

def get_random_mask(lb, ub, inclusive=False):
    while True:
        mask = random.uniform(ub, lb)
        if inclusive:
            return mask
        else:
            if lb < mask < ub:
                return mask

# def h_r(vec1: CKKSVector, k, encrypt_map, user_profile) -> List[List[Vertex]]:
#     P = []
#     scores = []
#     vec1_uri = list(encrypt_map_server.keys())[list(encrypt_map_server.values()).index(vec1)]
#     list_edges = user_profile.get_edges(get_vertex_object(vec1_uri))
#     for edge in list_edges:
#         p = [edge.v1, edge.v2]
#         chosen_edge = edge
#         while True:
#             if chosen_edge.v2.outward_degree == 0:
#                 break
#
#             candidate_edges = user_profile.get_edges(chosen_edge.v2)
#             prediction = predictor.predict(chosen_edge.label, [word.label for word in candidate_edges])
#             if prediction == predictor.tokenizer.eos_token:
#                 break
#
#             for cand in candidate_edges:
#                 if cand.label == prediction:
#                     chosen_edge = cand
#
#             if chosen_edge.v2 in p:
#                 break
#
#             p.append(chosen_edge.v2)
#
#         P.append(p)
#         scores.append(r_p(p))
#
#     sorted_paths = [k for _, k in sorted(zip(scores, P), reverse=True, key=lambda pair: pair[0])]
#
#     return sorted_paths[:k]
#
# def r_p(path: List[Vertex]) -> float:
#     product = 1
#
#     for i in range(0, len(path) - 1):
#         product = product * (1.0 / path[i].outward_degree)
#
#     return product
#
# def get_vertex_object(v_uri: str, vertices: List[Vertex]) -> Vertex | None:
#     for v in vertices:
#         if v.uri == v_uri:
#             return v
#     return None