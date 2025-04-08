import tenseal as ts
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict
import math

from tenseal import CKKSVector

from graph import Entity
import os


class LLMHelper:
    def __init__(self, model_name: str = "model/all-MiniLM-L12-v2"):
        """
        Initialize the SentenceTransformer model and set up an empty embedding map.
        """
        os.environ['CURL_CA_BUNDLE'] = ''
        self.model = SentenceTransformer(model_name)
        self.embed_map: Dict[str, np.ndarray] = {}
        self.encrypt_map: Dict[str, CKKSVector] = {}
    
    def encode_embedding(self, entities: List[Entity]) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for each entity using its label and store the results in a map,
        with the entity URI as the key.
        """
        embed_map: Dict[str, np.ndarray] = {}
        for e in entities:
            # Generate embedding using the entity's label.
            embedding = self.model.encode(e.get_label())
            embed_map[e.uri] = embedding
        self.embed_map = embed_map  # save the embedding map for later use
        return embed_map

    def encode_path(self, p1: List[Entity]):
        p1_sentence = ' '.join([e.label for e in p1])
        return self.model.encode(p1_sentence)

    def encrypt_path(self, context: ts.Context, p1_embedding: np.ndarray, normalize: bool = True) -> CKKSVector:
        if normalize:
            p1_embedding = p1_embedding / np.linalg.norm(p1_embedding)
        return ts.ckks_vector(context, p1_embedding)

    def encrypt_embeddings(self, context: ts.Context, normalize: bool = True) -> Dict[str, CKKSVector]:
        for (uri, embedding) in self.embed_map.items():
            if uri not in self.encrypt_map:
                if normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                encryption = ts.ckks_vector(context, embedding)
                self.encrypt_map[uri] = encryption
        return self.encrypt_map

    def _cosine_similarity_pltxt(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute the cosine similarity between two plaintext embeddings.
        """
        similarity = 1.0 - cosine(embedding1, embedding2)
        return similarity

    def _cosine_similarity_secure(self, normalized_v1: CKKSVector, normalized_v2: CKKSVector) -> CKKSVector:
        """
        Compute a "secure" cosine similarity between two encrypted embeddings.
        This is a placeholder; the exact implementation depends on your homomorphic encryption scheme.
        """
        return (normalized_v1 * normalized_v2).sum()

    def plntxt_vertex_similarity_check(self, uri1: str, uri2: str, sigma: float) -> bool:
        """
        Check if the cosine similarity between the plaintext embeddings of the two vertices
        (identified by their URIs) is at least sigma.
        """
        # Ensure that embeddings for both URIs exist.
        assert uri1 in self.embed_map and uri2 in self.embed_map, "Embeddings not found for one or both URIs."
        embed1 = self.embed_map[uri1]
        embed2 = self.embed_map[uri2]
        return sigma <= self._cosine_similarity_pltxt(embed1, embed2)
    def secure_vertex_similarity_check(self, encrypted_v1: CKKSVector, encrypted_v2: CKKSVector, sigma: float) -> bool:
        """
        Check if the cosine similarity between the encrypted embeddings of the two vertices
        (identified by their URIs) is at least sigma.
        """
        return self._cosine_similarity_secure(encrypted_v1, encrypted_v2).decrypt()[0] >= sigma