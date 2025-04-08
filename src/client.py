import socket
import struct

import numpy as np
from encryption import Encryption
from tenseal import Context

from graph import Vertex, Graph, Entity
import pickle
from llm import LLMHelper
from typing import List, Dict

from tenseal import CKKSVector
import tenseal as ts
import threading
from predictor import LLMPredictor


HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

v1: Vertex = Vertex('amy/v1', 'v1')
v2: Vertex = Vertex('amy/v2', 'Amy')
v3: Vertex = Vertex('amy/v3', 'Charlie')
v4: Vertex = Vertex('amy/v4', 'New York')
v5: Vertex = Vertex('amy/v5', 'Soccer')

v6: Vertex = Vertex('amy/v6', 'Red')

v7: Vertex = Vertex('amy/v7', 'House')
v8: Vertex = Vertex('amy/v8', 'Condo')
v9: Vertex = Vertex('amy/v9', '2')
v10: Vertex = Vertex('amy/v10', '3')

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

edge_v1_v2 = user_profile.add_edge(v1, v2, 'is named')
edge_v1_v3 = user_profile.add_edge(v1, v3, 'friends with')
edge_v1_v4 = user_profile.add_edge(v1, v4, 'born in')
edge_v1_v5 = user_profile.add_edge(v1, v5, 'plays')

edge_v3_v6 = user_profile.add_edge(v3, v6, 'likes color')

edge_v3_v7 = user_profile.add_edge(v3, v7, 'bought')
edge_v7_v8 = user_profile.add_edge(v7, v8, 'is a')
edge_v8_v9 = user_profile.add_edge(v8, v9, 'num beds')
edge_v8_v10 = user_profile.add_edge(v8, v10, 'num baths')

model: LLMHelper = LLMHelper()
predictor: LLMPredictor = LLMPredictor(model_name='model/TinyLlama-1.1B-Chat-v1.0')
encryption_helper: Encryption = Encryption()
context: Context = encryption_helper.get_context()
vertices: List[Vertex] = user_profile.vertices
vertex: List[Vertex] = [v3]
embedding_map: Dict[str, np.ndarray] = model.encode_embedding(vertices)
encrypt_map_client: Dict[str, CKKSVector] = model.encrypt_embeddings(context, normalize = True)
print(encrypt_map_client)
serialized_map = {}
epsilon = 0.0001

def decrypt_vector(encrypted_vector_bytes):
    """Decrypts an encrypted CKKS vector."""
    encrypted_vector = ts.ckks_vector_from(context, encrypted_vector_bytes)
    return encrypted_vector.decrypt()[0]

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
    return msg_type, msg

def request_handler(conn):
    while True:
        msg_type, msg = receive_full_message(conn)
        if not msg:
            break  # Connection closed

        if msg_type == 1:
            encrypted_vector_bytes = pickle.loads(msg)
            decrypted_values = decrypt_vector(encrypted_vector_bytes)

            if decrypted_values < 0 - epsilon:
                response = 0
            else:
                response = 1
            # Send back the decrypted values
            conn.sendall(struct.pack("!I", response))
        elif msg_type == 2:
            encrypted_vector_map = pickle.loads(msg)
            client_vec_uri = encrypted_vector_map["Vector"]
            k = struct.unpack("!I", encrypted_vector_map["K"])[0]
            client_vec = encrypt_map_client[client_vec_uri]
            paths = h_r(client_vec, k)
            for path in paths:
                print(f'Path: {[x.label for x in path]}')
            paths_encrypted = [[encrypt_map_client[x.uri] for x in path] for path in paths]
            uris = [[x.uri for x in path] for path in paths]

            paths_serialized = [[v.serialize() for v in path] for path in paths_encrypted]

            paths_uris_map_serialized = pickle.dumps({"URIs": uris, "Paths": paths_serialized})

            conn.sendall(struct.pack("!I", len(paths_uris_map_serialized)) + paths_uris_map_serialized)



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

def h_r(vec1: CKKSVector, k: int) -> List[List[Vertex]]:
    P = []
    scores = []
    vec1_uri = list(encrypt_map_client.keys())[list(encrypt_map_client.values()).index(vec1)]
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

serialized_encrypt_map_client: Dict[str, bytes] = {}

for uri, vec in encrypt_map_client.items():
    serialized_encrypt_map_client[uri] = vec.serialize()

serialized_context = encryption_helper.serialize_context()
serialized_map['Context'] = serialized_context
serialized_map['Vertex'] = {'amy/v3': serialized_encrypt_map_client['amy/v3']}

data = pickle.dumps(serialized_map)





with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    thread = threading.Thread(target=start_decryption_server, daemon=True).start()

    s.sendall(data)
    s.shutdown(socket.SHUT_WR)
    print('Sent Context and Vertex Embeddings')

    data = []
    while True:
        packet = s.recv(4096)
        if not packet: break
        data.append(packet)

    serialized_encrypt_map_server = pickle.loads(b"".join(data))

    if isinstance(serialized_encrypt_map_server, Dict):
        print('Received encryption')
    else:
        print('Data corrupted')

dot_product_mask: Dict[str, float] = {}

for uri, vec_bytes in serialized_encrypt_map_server.items():
    vec = ts.ckks_vector_from(context, vec_bytes)
    dot_product_mask[uri] = vec.decrypt()[0]


print(dot_product_mask)

# client_v3_encrypted_embeddings = encrypt_map_client['amy/v3']
# server_v3_encrypted_embeddings = encrypt_map_server['bob/v3']
#
# sigma = 0.5
# cosine_sim_greater_than_threshold = model.secure_vertex_similarity_check(client_v3_encrypted_embeddings, server_v3_encrypted_embeddings, sigma)
# print(f'Vertices cosine similarity above sigma ({sigma}): {cosine_sim_greater_than_threshold}')
#
# p1: List[Entity] = [v1, edge_v1_v3, v3]
# server_p1 = ts.ckks_vector_from(context, serialized_encrypt_map_server['Path'])
# client_p1 = model.encrypt_path(context, model.encode_path(p1))
#
# cosine_sim_greater_than_threshold = model.secure_vertex_similarity_check(server_p1, client_p1, sigma)
# print(f'Path cosine similarity above sigma ({sigma}): {cosine_sim_greater_than_threshold}')