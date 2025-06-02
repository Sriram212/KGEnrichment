import socket
import struct

import tenseal as ts
from tenseal import Context

from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from graph import Graph, Vertex, Entity, Edge
import pickle
import numpy as np
from typing import Dict, List, Tuple
from llm import LLMHelper
from predictor import LLMPredictor
from util import get_random_mask, merge_graphs, append_subgraph_at_uri, sum_values, merge_subgraph, write_table
# from graph_example_server import get_graph
from create_bob_graph import get_graph
import time
from visualize_graph import visualize_graph

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

# v1: Vertex = Vertex('bob/v1', 'v1')
# v2: Vertex = Vertex('bob/v2', 'Bob')
# v3: Vertex = Vertex('bob/v3', 'Charlie')
# v4: Vertex = Vertex('bob/v4', 'Minneapolis')
# v5: Vertex = Vertex('bob/v5', 'Soccer')
#
# v6: Vertex = Vertex('bob/v6', 'Basketball')
#
# v7: Vertex = Vertex('bob/v7', 'Chicago')
# v8: Vertex = Vertex('bob/v8', 'Illinois')
# v9: Vertex = Vertex('bob/v9', 'Springfield')
#
# v10: Vertex = Vertex('bob/v10', 'Red')
#
# v11: Vertex = Vertex('bob/v11', 'v11')
# v12: Vertex = Vertex('bob/v12', '2018')
# v13: Vertex = Vertex('bob/v13', 'Toyota')
# v14: Vertex = Vertex('bob/v14', 'Camry')
#
# user_profile: Graph = Graph()
#
# user_profile.add_vertex(v1)
# user_profile.add_vertex(v2)
# user_profile.add_vertex(v3)
# user_profile.add_vertex(v4)
# user_profile.add_vertex(v5)
#
# user_profile.add_vertex(v6)
# user_profile.add_vertex(v7)
# user_profile.add_vertex(v8)
# user_profile.add_vertex(v9)
# user_profile.add_vertex(v10)
#
# user_profile.add_vertex(v11)
# user_profile.add_vertex(v12)
# user_profile.add_vertex(v13)
# user_profile.add_vertex(v14)
#
# edge_v1_v2 = user_profile.add_edge(v1, v2, 'is named')
# edge_v1_v3 = user_profile.add_edge(v1, v3, 'friends with')
# edge_v1_v4 = user_profile.add_edge(v1, v4, 'born in')
# edge_v1_v5 = user_profile.add_edge(v1, v5, 'plays')
#
# edge_v3_v6 = user_profile.add_edge(v3, v6, 'plays')
#
# edge_v3_v7 = user_profile.add_edge(v3, v7, 'lives in')
# edge_v7_v8 = user_profile.add_edge(v7, v8, 'is in')
# edge_v8_v9 = user_profile.add_edge(v8, v9, 'capital is')
#
# edge_v3_v10 = user_profile.add_edge(v3, v10, 'likes color')
#
# edge_v3_v11 = user_profile.add_edge(v3, v11, 'drives')
#
# edge_v11_v12 = user_profile.add_edge(v11, v12, 'year')
# edge_v11_v13 = user_profile.add_edge(v11, v13, 'make')
# edge_v11_v14 = user_profile.add_edge(v11, v14, 'model')

times_dict = {}
bytes_sent_dict = {}
bytes_rec_list = []
hv_cache = {}

# v1: Vertex = Vertex('bob/v1', 'v1')
# v2: Vertex = Vertex('bob/v2', 'Bob')
# v3: Vertex = Vertex('bob/v3', 'Red')
# v4: Vertex = Vertex('bob/v4', 'Violin')
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

# user_profile = Graph()
#
# v1 = Vertex("bob/v1", "Bob")
# v2 = Vertex("bob/v2", "Amy")
#
# user_profile.add_vertex(v1)
# user_profile.add_vertex(v2)
#
# user_profile.add_edge(v1, v2, "friends with")

user_profile = get_graph()
visualize_graph(user_profile, 'server_before2.png')

start = time.time()
model: LLMHelper = LLMHelper()
end = time.time()
times_dict["Initialize Sentence Transformer"] = end-start

start = time.time()
predictor: LLMPredictor = LLMPredictor(model_name="model/TinyLlama-1.1B-Chat-v1.0")
end = time.time()
times_dict["Initialize LLM"] = end - start

vertices: List[Vertex] = user_profile.vertices

start = time.time()
embedding_map_server: Dict[str, np.ndarray] = model.encode_embedding(vertices)
end = time.time()
times_dict["Compute Embeddings"] = end - start
# p1: List[Entity] = [v1, edge_v1_v3, v3]
# p1_embedding = model.encode_path(p1)

mask = get_random_mask(1, 2, False)
sigma = 0.6
delta = 0.6

# cache = {}
ecache = {}

client_embed_map = {}

# Find the matching key
def find_key_by_value(dictionary, target_array):
    for key, value in dictionary.items():
        if np.array_equal(value, target_array):
            return key
    return None  # or raise an exception if needed

def get_vertex_object(v_uri: str) -> Vertex | None:
    for v in vertices:
        if v.uri == v_uri:
            return v
    return None

# def serialize_encryption_map(encrypted_map: Dict[str, CKKSVector]) -> Dict[str, bytes]:
#     result: Dict[str, bytes] = {}
#     for uri, encryption in encrypted_map.items():
#         result[uri] = encryption.serialize()
#     return result

def get_client_sub_graph(client_uri: str, decryption_socket: socket) -> Graph:
    client_uri_bytes = client_uri.encode()
    response_bytes = struct.pack("!I", len(client_uri_bytes)) + struct.pack("!I", 4) + client_uri_bytes

    if "Get Client Sub Graph" in bytes_sent_dict:
        bytes_sent_dict["Get Client Sub Graph"].append(len(response_bytes))
    else:
        bytes_sent_dict["Get Client Sub Graph"] = [len(response_bytes)]

    decryption_socket.sendall(response_bytes)
    response = receive_full_message(decryption_socket)
    return pickle.loads(response)


def h_v(vec1: np.ndarray, vec2: np.ndarray):
    start = time.time()

    if (vec1.tobytes(), vec2.tobytes()) in hv_cache:

        end = time.time()
        total_time = end - start

        if "Vertex Similarity" in times_dict:
            times_dict["Vertex Similarity"].append(total_time)
        else:
            times_dict["Vertex Similarity"] = [total_time]

        return hv_cache[(vec1.tobytes(), vec2.tobytes())]

    m_v = vec1.dot(vec2) - sigma
    response_bool = m_v >= 0
    end = time.time()
    total_time = end - start

    if "Vertex Similarity" in times_dict:
        times_dict["Vertex Similarity"].append(total_time)
    else:
        times_dict["Vertex Similarity"] = [total_time]

    hv_cache[(vec1.tobytes(), vec2.tobytes())] = response_bool

    return response_bool

def h_p(path1: np.ndarray, path1_len: int, path2: np.ndarray, path2_len: int) -> float:
    # m_p = ((path1.dot(path2) * (1.0/(path1_len + path2_len))) - delta) * mask
    start = time.time()
    m_p = (path1.dot(path2)) / (path1_len + path2_len)
    m_p = m_p - sigma
    # print(f"MP: {m_p.decrypt()[0]}")
    # print(f'Dot: {path1.dot(path2).decrypt()[0]}')

    return m_p

def r_p(path: List[Vertex]) -> float:
    product = 1

    for i in range(0, len(path) - 1):
        product = product * (1.0 / path[i].outward_degree)

    return product

def h_r(vec1: np.ndarray, k) -> Tuple[List[List[Vertex]], List[List[Edge]]]:
    start = time.time()
    P = []
    scores = []
    edges = []
    vec1_uri = find_key_by_value(embedding_map_server, vec1)
    # vec1_uri = list(embedding_map_server.keys())[list(embedding_map_server.values()).index(vec1)]
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

            p.append(chosen_edge.v2)
            e.append(chosen_edge)

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

def receive_full_message(conn):
    """Receives a message with a fixed-length header."""
    msg_len_data = conn.recv(4)  # Get the first 4 bytes (message length)

    if not msg_len_data:
        return None
    msg_len = struct.unpack("!I", msg_len_data)[0]  # Unpack message length

    total_bytes_rec = 4 + msg_len
    bytes_rec_list.append(total_bytes_rec)

    # Receive the full message
    msg = b""
    while len(msg) < msg_len:
        msg_packet = conn.recv(msg_len - len(msg))
        if not msg_packet:
            return None
        msg += msg_packet
    return msg


def para_match(vec1: np.ndarray, vec1_uri: str, vec2: np.ndarray, vec2_uri: str, delta, k, decryption_socket: socket) -> bool:
    start = time.time()
    # print(f"Vec2: {vec2_uri}")
    if not h_v(vec1, vec2):
        cache[(vec1_uri, vec2_uri)] = [False, []]
        return False

    vertex = user_profile.lookup(vec1_uri)
    if type(vertex) != Vertex:
        return False

    if vertex.outward_degree == 0:
        cache[(vec1_uri, vec2_uri)] = [True, []]
        return True

    cache[(vec1_uri, vec2_uri)] = [True, []]
    W = []
    sum = 0

    V_server = []
    V_client = []

    if vec1_uri not in ecache:
        paths, edges = h_r(vec1, k)
        # print(f'Edges: {[[x.label for x in edge] for edge in edges]}')
        # print(f'Paths: {[[x.label for x in path] for path in paths]}')
        server_paths: List[List[np.ndarray]] = [[embedding_map_server[x.uri] for x in path] for path in paths]
        server_uris = [[x.uri for x in path] for path in paths]
        server_lengths = [len(path) for path in paths]
        server_edges: List[np.ndarray] = [model.encode_path(x) for x in edges]

        for path in paths:
            uri = path[1].uri
            V_server.append((uri, embedding_map_server[uri]))
        ecache[vec1_uri] = V_server

    if vec2_uri not in ecache:
        k_serialized = struct.pack("!I", k)

        h_r_client_map = {"Vector": vec2_uri, "K": k_serialized}
        h_r_client_bytes = pickle.dumps(h_r_client_map)

        response_bytes = struct.pack("!I", len(h_r_client_bytes)) + struct.pack("!I", 2) + h_r_client_bytes

        if "Top-K Paths" in bytes_sent_dict:
            bytes_sent_dict["Top-K Paths"].append(len(response_bytes))
        else:
            bytes_sent_dict["Top-K Paths"] = [len(response_bytes)]

        decryption_socket.sendall(response_bytes)
        msg = receive_full_message(decryption_socket)

        paths_edges_uris_map = pickle.loads(msg)

        client_paths = []
        client_edges = []
        client_lengths = []
        client_uris = paths_edges_uris_map["URIs"]
        # print(f"CLIENT URIs: {client_uris}")
        for path in paths_edges_uris_map["Vectors"]:
            client_paths.append(path)

        for edge in paths_edges_uris_map["Edges"]:
            client_edges.append(edge)

        for length in paths_edges_uris_map["Length"]:
            client_lengths.append(length)

        for i in range(0, len(client_paths)):
            V_client.append((client_uris[i], client_paths[i]))

        ecache[vec2_uri] = V_client

    L = {}
    max_score = 0
    for s_index in range(len(V_server)):
        server_prime_uri, server_prime_vec = V_server[s_index]
        l_u_prime = []
        scores = []
        for c_index in range(len(V_client)):
            client_prime_uri, client_prime_vec = V_client[c_index]
            # print(f"Client URI: {client_prime_uri}, Server URI: {server_prime_uri}")
            if h_v(server_prime_vec, client_prime_vec):
                # print("HERE")
                l_u_prime.append((client_prime_uri, client_prime_vec))
                score = h_p(server_edges[s_index], server_lengths[s_index], client_edges[c_index], client_lengths[c_index])
                # print(f"Dot Product: {server_edges[s_index].dot(client_edges[c_index]).decrypt()[0]}")
                scores.append(score)
        sorted_l_u_prime = [k for _, k in sorted(zip(scores, l_u_prime), reverse=True, key=lambda pair: pair[0])]
        scores = sorted(scores)
        if len(scores) > 0:
            max_score += scores[0]
        # print(f"L_U': {sorted_l_u_prime}, Scores: {scores}")
        if len(sorted_l_u_prime) > 0:
            L[(server_prime_uri, server_prime_vec.tobytes())] = sorted_l_u_prime
            # L.append(sorted_l_u_prime)
    # print(L)
    # print(f"SCORE: {max_score}")

    if max_score < delta:
        cache[(vec1_uri, vec2_uri)] = [False, []]
        return False

    for server_prime_uri, server_prime_vec in V_server:
        for client_prime_uri, client_prime_vec in L[(server_prime_uri, server_prime_vec.tobytes())]:
            if (server_prime_uri, client_prime_uri) in cache:
                match = cache[(server_prime_uri, client_prime_uri)][0]
            else:
                match = para_match(server_prime_vec, server_prime_uri, client_prime_vec, client_prime_uri, delta, k, decryption_socket)
            if match:
                index = 0
                # p1_length = 0
                for path in server_paths:
                    if path[1] == server_prime_vec:
                        break
                    index += 1
                server_path = server_edges[index]
                p1_length = server_lengths[index]

                index = 0
                # p2_length = 0
                for path in client_paths:
                    if path == client_prime_vec:
                        break
                    index += 1
                p2_length = client_lengths[index]
                client_path = client_edges[index]
                sum += h_p(server_path, p1_length, client_path, p2_length)
                W.append((server_prime_uri, client_prime_uri))
                if sum > delta:
                    cache[(vec1_uri, vec2_uri)] = [True, W]
                    return True
                break

            index = 0
            # p1_length = 0
            for path in server_paths:
                if path[1] == server_prime_vec:
                    break
                index += 1
            server_path = server_edges[index]
            p1_length = server_lengths[index]

            index = 0
            # p2_length = 0
            for path in client_paths:
                if path == client_prime_vec:
                    break
                index += 1
            client_path = client_edges[index]
            p2_length = client_lengths[index]
            max_score -= h_p(server_path, p1_length, client_path, p2_length)

            for client_prime_n_uri, client_prime_n_vec in L[(server_prime_uri, server_prime_vec.tobytes())]:
                if client_prime_n_uri != client_prime_uri:
                    index = 0
                    # p2_length = 0
                    for path in client_paths:
                        if path == client_prime_n_vec:
                            break
                        index += 1
                    client_path = client_edges[index]
                    p2_length = client_lengths[index]

                    max_score += h_p(server_path, p1_length, client_path, p2_length)

            if max_score < delta:
                break

    cache[(vec1_uri, vec2_uri)] = [False, []]

    for server_p_uri, client_p_uri in cache:
        if (vec1_uri, vec2_uri) in cache[(server_p_uri, client_p_uri)][1]:
            del cache[(server_p_uri, client_p_uri)]
            para_match(embedding_map_server[server_p_uri], server_p_uri, client_embed_map[client_p_uri], client_p_uri, delta, k, decryption_socket)

    end = time.time()
    total_time = end - start
    if "ParaMatch" in times_dict:
        times_dict["ParaMatch"].append(total_time)
    else:
        times_dict["ParaMatch"] = [total_time]
    return False


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print('Ready to Connect')
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        start_time = time.time()
        data = []
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data.append(packet)

        bytes_rec_list.append(len(b"".join(data)))

        serialized_embedding_map_client = pickle.loads(b"".join(data))

        if isinstance(serialized_embedding_map_client, Dict):
            print('Received encryption')
        else:
            print('Data corrupted')

        # context: Context = ts.context_from(data=serialized_encrypt_map_client['Context'])
        vertex_embedding_client = None
        uri_client = None

        for uri, vec in serialized_embedding_map_client['Vertices'].items():
            uri_client = uri
            vertex_embedding_client = vec
            client_embed_map[uri_client] = vertex_embedding_client

        # paths_client = [[ts.ckks_vector_from(context, vec) for vec in path] for path in serialized_encrypt_map_client['Paths']]

        # encryption_start_time = time.time()
        # # encrypt_map_server: Dict[str, CKKSVector] = model.encrypt_embeddings(context, normalize = True)
        # encryption_end_time = time.time()
        # times_dict["Encryption"] = encryption_end_time - encryption_start_time


        # vertex_dot_product_map: Dict[str, CKKSVector] = {}

        serialized_map_server = {}

        decryption_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        decryption_socket.connect((HOST, PORT + 10))

        #VParaMatch
        v_para_match_start = time.time()
        # PI = {}
        # C = {}
        # for uri_server, vec_server in encrypt_map_server.items():
        #     PI[uri_server] = []
        #     C[uri_server] = []
        #     cache = {}
        #     for uri_client, vec_client in client_encrypt_map.items():
        #         if h_v(vec_server, vec_client, decryption_socket):
        #             if (uri_server, uri_client) in cache and cache[(uri_server, uri_client)][0] == True:
        #                 PI[uri_server].append((uri_server, uri_client))
        #             else:
        #                 match = para_match(vec_server, uri_server, vec_client, uri_client, delta, 3, decryption_socket)
        #                 # print(f"Server URI: {uri_server}, Client URI: {uri_client}, Match: {match}")
        #                 if match:
        #                     PI[uri_server].append(uri_client)
        #     print("Finished for Server Node")

        PI = {}
        C = {}

        for uri_server, vec_server in embedding_map_server.items():
            PI[uri_server] = []
            C[uri_server] = []
            cache = {}


            # worker for a single client‐vector pair
            def check_client(uri_client, vec_client):
                # first h_v check
                if not h_v(vec_server, vec_client):
                    return None

                # cache hit?
                if cache.get((uri_server, uri_client), (False,))[0]:
                    return uri_client

                # do the expensive match
                match = para_match(
                    vec_server,
                    uri_server,
                    vec_client,
                    uri_client,
                    delta,
                    3,
                    decryption_socket
                )
                cache[(uri_server, uri_client)] = (bool(match),)

                return uri_client if match else None


            # parallelize the inner loop
            with ThreadPoolExecutor(max_workers=min(len(client_embed_map), os.cpu_count() or 4)) as pool:
                futures = {
                    pool.submit(check_client, uri_client, vec_client): uri_client
                    for uri_client, vec_client in client_embed_map.items()
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        PI[uri_server].append(result)
            print("Server Node Complete")


                    # product_mask = (vec_client.dot(vec_server) - sigma) * mask
                    # match = para_match(vec_server, uri_server, vec_client, uri_client, delta, 3, decryption_socket)
                    # print(f"URI: {uri}, Match: {match}")
                    # vertex_dot_product_map[uri] = product_mask
                    # serialized_map_server[uri] = product_mask.serialize()

        # print(PI)
        # temp = {'bob/v2': 1, 'bob/v3': 1, 'bob/v1': 1}
        # temp_ordered = dict(sorted(temp.items(), key = lambda item: user_profile.lookup(item[0]).outward_degree, reverse=True))
        # print(temp_ordered)
        PI_ordered = dict(sorted(PI.items(), key = lambda item: user_profile.lookup(item[0]).outward_degree, reverse=True))
        v_para_match_end = time.time()
        times_dict["VParaMatch"] = v_para_match_end - v_para_match_start


        #Enrichment
        print(f"PI Ordered: {PI_ordered}")
        enrichment_start_time = time.time()
        graphs_to_send = []
        for uri_server in PI_ordered:
            graph_map = {}
            # server_sub_graph = user_profile.extract_lineage_set(user_profile.lookup(uri_server))
            # server_sub_graph_bytes = pickle.dumps(server_sub_graph)
            for client_uri in PI_ordered[uri_server]:
                client_sub_graph = get_client_sub_graph(client_uri, decryption_socket)
                client_vertex = client_sub_graph.lookup(client_uri)
                server_vertex = user_profile.lookup(uri_server)
                merge_subgraph(client_sub_graph, client_vertex, user_profile, server_vertex)
                # merged_graph = merge_graphs(server_sub_graph, client_sub_graph)
                # merged_graph.print_graph()
                # append_subgraph_at_uri(user_profile, merged_graph, uri_server)
                # user_profile.print_graph()

                #Send Client server sub graph
                # data_map = {"Subgraph": server_sub_graph, "URI": client_uri}
                # data_bytes = pickle.dumps(data_map)
                # response_bytes = struct.pack("!I", len(data_bytes)) + struct.pack("!I", 5) + data_bytes
                #
                # if "Sub Graph" in bytes_sent_dict:
                #     bytes_sent_dict["Sub Graph"].append(len(response_bytes))
                # else:
                #     bytes_sent_dict["Sub Graph"] = [len(response_bytes)]
                #
                # decryption_socket.sendall(response_bytes)
                # graph_map[client_uri] = server_sub_graph

            # graphs_to_send.append(graph_map)


        # data = pickle.dumps(graphs_to_send)


        # conn.sendall(data)
        # data = pickle.dumps(serialized_map_server)
        enrichment_end_time = time.time()
        end_time = time.time()
        total_time = end_time - start_time
        times_dict["Enrichment"] = enrichment_end_time - enrichment_start_time
        data = struct.pack("!I", 0)
        bytes_sent_dict["End"] = 4
        conn.sendall(data)
        total_bytes_sent = sum_values(bytes_sent_dict)
        total_bytes_received = sum(bytes_rec_list)
        print(f"Total Bytes Sent: {total_bytes_sent}")
        print(f"Total Bytes Received: {total_bytes_received}")
        print(f"Times: {times_dict}")
        print(f"Total Time: {total_time}\nEND")
        # measurements = [(total_bytes_sent, total_bytes_received, total_time)]
        # write_table(measurements, "results.txt", False)
        user_profile.print_graph()
        visualize_graph(user_profile, "server_after2.png")