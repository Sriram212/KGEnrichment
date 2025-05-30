import socket
import struct
import time

import numpy as np
from encryption import Encryption
from tenseal import Context

from graph import Vertex, Graph, Entity, Edge
import pickle
from llm import LLMHelper
from typing import List, Dict, Tuple

# from graph_example_client import get_graph
from create_amy_graph import get_graph

from tenseal import CKKSVector
import tenseal as ts
import threading
from predictor import LLMPredictor
from util import get_random_mask, merge_graphs, append_subgraph_at_uri, sum_values

from visualize_graph import visualize_graph

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

# v1: Vertex = Vertex('amy/v1', 'v1')
# v2: Vertex = Vertex('amy/v2', 'Amy')
# v3: Vertex = Vertex('amy/v3', 'Charlie')
# v4: Vertex = Vertex('amy/v4', 'New York')
# v5: Vertex = Vertex('amy/v5', 'Soccer')
#
# v6: Vertex = Vertex('amy/v6', 'Red')
#
# v7: Vertex = Vertex('amy/v7', 'House')
# v8: Vertex = Vertex('amy/v8', 'Condo')
# v9: Vertex = Vertex('amy/v9', '2')
# v10: Vertex = Vertex('amy/v10', '3')
#
# user_profile: Graph = Graph()
#
# user_profile.add_vertex(v1)
# user_profile.add_vertex(v2)
# user_profile.add_vertex(v3)
# user_profile.add_vertex(v4)
# user_profile.add_vertex(v5)
# user_profile.add_vertex(v6)
# user_profile.add_vertex(v7)
# user_profile.add_vertex(v8)
# user_profile.add_vertex(v9)
# user_profile.add_vertex(v10)
#
# edge_v1_v2 = user_profile.add_edge(v1, v2, 'is named')
# edge_v1_v3 = user_profile.add_edge(v1, v3, 'friends with')
# edge_v1_v4 = user_profile.add_edge(v1, v4, 'born in')
# edge_v1_v5 = user_profile.add_edge(v1, v5, 'plays')
#
# edge_v3_v6 = user_profile.add_edge(v3, v6, 'likes color')
#
# edge_v3_v7 = user_profile.add_edge(v3, v7, 'bought')
# edge_v7_v8 = user_profile.add_edge(v7, v8, 'is a')
# edge_v8_v9 = user_profile.add_edge(v8, v9, 'num beds')
# edge_v8_v10 = user_profile.add_edge(v8, v10, 'num baths')

# v1: Vertex = Vertex('amy/v1', 'v1')
# v2: Vertex = Vertex('amy/v2', 'Bob')
# v3: Vertex = Vertex('amy/v3', 'Red')
# v4: Vertex = Vertex('amy/v4', 'Soccer')
#
# user_profile: Graph = Graph()
#
# user_profile.add_vertex(v1)
# user_profile.add_vertex(v2)
# user_profile.add_vertex(v3)
# user_profile.add_vertex(v4)
#
# edge_v1_v2: Edge = user_profile.add_edge(v1, v2, 'is named')
# edge_v1_v3: Edge = user_profile.add_edge(v1, v3, 'likes color')
# edge_v1_v4: Edge = user_profile.add_edge(v1, v4, 'plays')

user_profile = get_graph()

times_dict = {}
bytes_sent_dict = {}
bytes_rec_list = []

start = time.time()
model: LLMHelper = LLMHelper()
end = time.time()
times_dict["Initialize Sentence Transformer"] = end-start

start = time.time()
predictor: LLMPredictor = LLMPredictor(model_name="model/TinyLlama-1.1B-Chat-v1.0")
end = time.time()
times_dict["Initialize LLM"] = end - start

encryption_helper: Encryption = Encryption()
vertices: List[Vertex] = user_profile.vertices

start = time.time()
embedding_map: Dict[str, np.ndarray] = model.encode_embedding(vertices)
end = time.time()
times_dict["Compute Embeddings"] = end - start
start = time.time()
# encrypt_map_client: Dict[str, CKKSVector] = model.encrypt_embeddings(context, normalize = True)
end = time.time()
times_dict["Encryption"] = end - start
# print(encrypt_map_client)
serialized_map = {}
epsilon = 0.01
mask = get_random_mask(1, 2, False)

# def decrypt_vector(encrypted_vector_bytes):
#     """Decrypts an encrypted CKKS vector."""
#     encrypted_vector = ts.ckks_vector_from(context, encrypted_vector_bytes)
#     return encrypted_vector.decrypt()[0]

# Find the matching key
def find_key_by_value(dictionary, target_array):
    for key, value in dictionary.items():
        if np.array_equal(value, target_array):
            return key
    return None  # or raise an exception if needed

def receive_full_message(conn):
    """Receives a message with a fixed-length header."""
    msg_len_data = conn.recv(4)  # Get the first 4 bytes (message length)
    if not msg_len_data:
        return None
    msg_len = struct.unpack("!I", msg_len_data)[0]  # Unpack message length

    msg_type_data = conn.recv(4)
    if not msg_type_data:
        return None
    msg_type = struct.unpack("!I", msg_type_data)[0]

    # Receive the full message
    msg = b""
    while len(msg) < msg_len:
        msg_packet = conn.recv(msg_len - len(msg))
        if not msg_packet:
            return None
        msg += msg_packet

    total_bytes_rec = 8 + msg_len
    bytes_rec_list.append(total_bytes_rec)

    return msg_type, msg

def request_handler(conn):
    while True:
        msg_type, msg = receive_full_message(conn)
        if not msg:
            break  # Connection closed
        elif msg_type == 2:
            encrypted_vector_map = pickle.loads(msg)
            client_vec_uri = encrypted_vector_map["Vector"]
            k = struct.unpack("!I", encrypted_vector_map["K"])[0]
            client_vec = embedding_map[client_vec_uri]
            paths, edges = h_r(client_vec, k)
            # for path in paths:
            #     print(f'Path: {[x.label for x in path]}')
            # for edge in edges:
            #     print(f'Edge: {[x.label for x in edge]}')

            # edges_vectors = [model.encode_path(edge) for edge in edges]

            paths_vectors_embeddings = [[embedding_map[x.uri] for x in path] for path in paths]
            path_lengths = [len(path) for path in paths]
            edges_vectors_embedding = [model.encode_path(edge) for edge in edges]

            # print(f'Length of edges vectors encrypted: {len(edges_vectors_encrypted)}')

            uris = [path[1].uri for path in paths]

            paths_serialized = [path[1] for path in paths_vectors_embeddings]
            # edges_serialized = [edge.serialize() for edge in edges_vectors_encrypted]
            # path_length_serialized = [length.serialize() for length in path_length_encrypted]

            paths_uris_edges_map_serialized = pickle.dumps({"URIs": uris, "Vectors": paths_serialized, "Edges": edges_vectors_embedding, "Length": path_lengths})
            client_top_k_paths_bytes = struct.pack("!I", len(paths_uris_edges_map_serialized)) + paths_uris_edges_map_serialized

            if "Top-K Paths" in bytes_sent_dict:
                bytes_sent_dict["Top-K Paths"].append(len(client_top_k_paths_bytes))
            else:
                bytes_sent_dict["Top-K Paths"] = [len(client_top_k_paths_bytes)]

            conn.sendall(client_top_k_paths_bytes)

        elif msg_type == 4:
            uri = msg.decode()
            sub_graph = user_profile.extract_lineage_set(user_profile.lookup(uri))
            sub_graph_bytes = pickle.dumps(sub_graph)
            response_bytes = struct.pack("!I", len(sub_graph_bytes)) + sub_graph_bytes

            if "Sub Graph" in bytes_sent_dict:
                bytes_sent_dict["Sub Graph"].append(len(response_bytes))
            else:
                bytes_sent_dict["Sub Graph"] = [len(response_bytes)]

            conn.sendall(response_bytes)
        elif msg_type == 5: #Enrichment
            enrichment_start_time = time.time()
            server_sub_graph_uri_map = pickle.loads(msg)
            uri = server_sub_graph_uri_map['URI']
            server_sub_graph = server_sub_graph_uri_map['Subgraph']
            client_sub_graph = user_profile.extract_lineage_set(user_profile.lookup(uri))

            merged_graph = merge_graphs(server_sub_graph, client_sub_graph)
            # merged_graph.print_graph()
            append_subgraph_at_uri(user_profile, merged_graph, uri)
            # user_profile.print_graph()
            enrichment_end_time = time.time()
            enrichment_total_time = enrichment_end_time - enrichment_start_time

            if "Enrichment" in times_dict:
                times_dict["Enrichment"].append(enrichment_total_time)
            else:
                times_dict["Enrichment"] = [enrichment_total_time]



def start_decryption_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT + 10))
    server_socket.listen(1)
    print("Client is waiting for server connection...")

    conn, addr = server_socket.accept()
    print(f"Connected to server: {addr}")

    request_handler(conn)

    conn.close()
    server_socket.close()

def get_vertex_object(v_uri: str) -> Vertex | None:
    for v in vertices:
        if v.uri == v_uri:
            return v
    return None

def r_p(path: List[Vertex]) -> float:
    product = 1

    for i in range(0, len(path) - 1):
        product = product * (1.0 / path[i].outward_degree)

    return product

def h_r(vec1: np.ndarray, k: int) -> Tuple[List[List[Vertex]], List[List[Edge]]]:
    start = time.time()
    P = []
    scores = []
    edges = []
    vec1_uri = find_key_by_value(embedding_map, vec1)
    # vec1_uri = list(embedding_map.keys())[list(embedding_map.values()).index(vec1)]
    list_edges = user_profile.get_edges(get_vertex_object(vec1_uri))
    for edge in list_edges:
        p = [edge.v1, edge.v2]
        e = [edge]
        chosen_edge = edge
        while True:
            if chosen_edge.v2.outward_degree == 0:
                break

            candidate_edges = user_profile.get_edges(chosen_edge.v2)
            prediction = predictor.predict(chosen_edge.label, [word.label for word in candidate_edges])
            if prediction == predictor.tokenizer.eos_token:
                break

            for cand in candidate_edges:
                if cand.label == prediction:
                    chosen_edge = cand

            if chosen_edge.v2 in p:
                break

            e.append(chosen_edge)
            p.append(chosen_edge.v2)

        edges.append(e)
        P.append(p)
        scores.append(r_p(p))

    sorted_paths = [k for _, k in sorted(zip(scores, P), reverse=True, key=lambda pair: pair[0])]
    sorted_edges = [k for _, k in sorted(zip(scores, edges), reverse=True, key=lambda pair: pair[0])]
    end = time.time()
    total_time = end - start
    if "Top-K Paths" in times_dict:
        times_dict["Top-K Paths"].append(total_time)
    else:
        times_dict["Top-K Paths"] = [total_time]
    return sorted_paths[:k], sorted_edges[:k]

serialized_map['Vertices'] = embedding_map

data = pickle.dumps(serialized_map)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    thread = threading.Thread(target=start_decryption_server, daemon=True).start()

    bytes_sent_dict["Context and Vertices"] = len(data)

    s.sendall(data)
    s.shutdown(socket.SHUT_WR)
    print('Sent Context and Vertex Embeddings')

    s.settimeout(500)
    data = s.recv(4)
    bytes_rec_list.append(4)
    msg_type = struct.unpack("!I", data)[0]
    print(f"Total Time: {sum_values(times_dict)}")
    print(f"Total Bytes Sent: {sum_values(bytes_sent_dict)}")
    print(f"Total Bytes Received: {sum(bytes_rec_list)}")
    # user_profile.print_graph()
    # visualize_graph(user_profile, "client.png")
    print("END")
