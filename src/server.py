import socket
import struct

import tenseal as ts
from tenseal import Context

from graph import Graph, Vertex, Entity, Edge
import pickle
import numpy as np
from typing import Dict, List
from llm import LLMHelper
from tenseal import CKKSVector
from predictor import LLMPredictor
from util import get_random_mask

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

v1: Vertex = Vertex('bob/v1', 'v1')
v2: Vertex = Vertex('bob/v2', 'Bob')
v3: Vertex = Vertex('bob/v3', 'Charlie')
v4: Vertex = Vertex('bob/v4', 'Minneapolis')
v5: Vertex = Vertex('bob/v5', 'Soccer')

v6: Vertex = Vertex('bob/v6', 'Basketball')

v7: Vertex = Vertex('bob/v7', 'Chicago')
v8: Vertex = Vertex('bob/v8', 'Illinois')
v9: Vertex = Vertex('bob/v9', 'Springfield')

v10: Vertex = Vertex('bob/v10', 'Violin')

v11: Vertex = Vertex('bob/v11', 'v11')
v12: Vertex = Vertex('bob/v12', '2018')
v13: Vertex = Vertex('bob/v13', 'Toyota')
v14: Vertex = Vertex('bob/v14', 'Camry')

user_profile: Graph = Graph()

user_profile.add_vertex(v1)
user_profile.add_vertex(v2)
user_profile.add_vertex(v3)
user_profile.add_vertex(v4)
user_profile.add_vertex(v5)

user_profile.add_vertex(v6)
user_profile.add_vertex(v7)
user_profile.add_vertex(v8)
user_profile.add_vertex(v9)
user_profile.add_vertex(v10)

user_profile.add_vertex(v11)
user_profile.add_vertex(v12)
user_profile.add_vertex(v13)
user_profile.add_vertex(v14)

edge_v1_v2 = user_profile.add_edge(v1, v2, 'is named')
edge_v1_v3 = user_profile.add_edge(v1, v3, 'friends with')
edge_v1_v4 = user_profile.add_edge(v1, v4, 'born in')
edge_v1_v5 = user_profile.add_edge(v1, v5, 'plays')

edge_v3_v6 = user_profile.add_edge(v3, v6, 'plays')

edge_v3_v7 = user_profile.add_edge(v3, v7, 'lives in')
edge_v7_v8 = user_profile.add_edge(v7, v8, 'is in')
edge_v8_v9 = user_profile.add_edge(v8, v9, 'capital is')

edge_v3_v10 = user_profile.add_edge(v3, v10, 'plays')

edge_v3_v11 = user_profile.add_edge(v3, v11, 'drives')

edge_v11_v12 = user_profile.add_edge(v11, v12, 'year')
edge_v11_v13 = user_profile.add_edge(v11, v13, 'make')
edge_v11_v14 = user_profile.add_edge(v11, v14, 'model')

model: LLMHelper = LLMHelper()
predictor: LLMPredictor = LLMPredictor(model_name="model/TinyLlama-1.1B-Chat-v1.0")
vertices: List[Vertex] = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14]
embedding_map_server: Dict[str, np.ndarray] = model.encode_embedding(vertices)
p1: List[Entity] = [v1, edge_v1_v3, v3]
p1_embedding = model.encode_path(p1)

mask = get_random_mask(1, 2, False)
sigma = 1.0

cache = {}
ecache = {}

def get_vertex_object(v_uri: str) -> Vertex | None:
    for v in vertices:
        if v.uri == v_uri:
            return v
    return None

def serialize_encryption_map(encrypted_map: Dict[str, CKKSVector]) -> Dict[str, bytes]:
    result: Dict[str, bytes] = {}
    for uri, encryption in encrypted_map.items():
        result[uri] = encryption.serialize()
    return result


def h_v(vec1: CKKSVector, vec2: CKKSVector, decryption_socket: socket):
    m_v = (vec1.dot(vec2) - sigma) * mask
    m_v_bytes = pickle.dumps(m_v.serialize())
    decryption_socket.sendall(struct.pack("!I", len(m_v_bytes)) + struct.pack("!I", 1) + m_v_bytes)

    response = decryption_socket.recv(4)
    if not response:
        return None

    response_bool = bool(struct.unpack("!I", response)[0])

    return response_bool

def r_p(path: List[Vertex]) -> float:
    product = 1

    for i in range(0, len(path) - 1):
        product = product * (1.0 / path[i].outward_degree)

    return product

def h_r(vec1: CKKSVector, k) -> List[List[Vertex]]:
    P = []
    scores = []
    vec1_uri = list(encrypt_map_server.keys())[list(encrypt_map_server.values()).index(vec1)]
    list_edges = user_profile.get_edges(get_vertex_object(vec1_uri))
    for edge in list_edges:
        p = [edge.v1, edge.v2]
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

        P.append(p)
        scores.append(r_p(p))

    sorted_paths = [k for _, k in sorted(zip(scores, P), reverse=True, key=lambda pair: pair[0])]

    return sorted_paths[:k]

def receive_full_message(conn):
    """Receives a message with a fixed-length header."""
    msg_len_data = conn.recv(4)  # Get the first 4 bytes (message length)
    if not msg_len_data:
        return None
    msg_len = struct.unpack("!I", msg_len_data)[0]  # Unpack message length

    # Receive the full message
    msg = b""
    while len(msg) < msg_len:
        msg_packet = conn.recv(msg_len - len(msg))
        if not msg_packet:
            return None
        msg += msg_packet
    return msg


def para_match(vec1: CKKSVector, vec1_uri: str, vec2: CKKSVector, vec2_uri: str, delta, k, decryption_socket: socket):
    if not h_v(vec1, vec2, decryption_socket):
        cache[(vec1, vec2)] = [False, []]
        return False

    vertex = user_profile.lookup(vec1_uri)
    if type(vertex) != Vertex:
        return False

    if vertex.outward_degree == 0:
        cache[(vec1, vec2)] = [True, []]
        return True

    cache[(vec1, vec2)] = [True, []]
    W = []
    sum = 0

    if vec1 not in ecache:
        paths = h_r(vec1, k)
        paths_encrypted: List[List[CKKSVector]] = [[encrypt_map_server[x.uri] for x in path] for path in paths]

    if vec2 not in ecache:
        k_serialized = struct.pack("!I", k)

        h_r_client_map = {"Vector": vec2_uri, "K": k_serialized}
        h_r_client_bytes = pickle.dumps(h_r_client_map)

        decryption_socket.sendall(struct.pack("!I", len(h_r_client_bytes)) + struct.pack("!I", 2) + h_r_client_bytes)
        msg = receive_full_message(decryption_socket)

        paths_uris_map = pickle.loads(msg)

        client_paths = []
        client_uris = paths_uris_map["URIs"]
        for path in paths_uris_map["Paths"]:
            client_paths.append([ts.ckks_vector_from(context, x) for x in path])

        print(f"Client URIs: {client_uris}\nClient Paths: {client_paths}")


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print('Ready to Connect')
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        data = []
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data.append(packet)

        serialized_encrypt_map_client = pickle.loads(b"".join(data))

        if isinstance(serialized_encrypt_map_client, Dict):
            print('Received encryption')
        else:
            print('Data corrupted')

        context: Context = ts.context_from(data=serialized_encrypt_map_client['Context'])
        vertex_embedding_client = None
        uri_client = None

        for uri, vec in serialized_encrypt_map_client['Vertex'].items():
            uri_client = uri
            vertex_embedding_client = ts.ckks_vector_from(context, vec)

        # paths_client = [[ts.ckks_vector_from(context, vec) for vec in path] for path in serialized_encrypt_map_client['Paths']]


        encrypt_map_server: Dict[str, CKKSVector] = model.encrypt_embeddings(context, normalize = True)

        vertex_dot_product_map: Dict[str, CKKSVector] = {}

        serialized_map_server = {}

        decryption_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        decryption_socket.connect((HOST, PORT + 10))

        for uri, vec in encrypt_map_server.items():
            product_mask = (vertex_embedding_client.dot(vec) - sigma) * mask
            para_match(vec, uri, vertex_embedding_client, uri_client, 0, 3, decryption_socket)
            vertex_dot_product_map[uri] = product_mask
            serialized_map_server[uri] = product_mask.serialize()

        data = pickle.dumps(serialized_map_server)
        conn.sendall(data)