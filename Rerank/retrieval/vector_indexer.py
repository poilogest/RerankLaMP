import faiss
import numpy as np
from models.encoder import Encoder

class VectorIndexer:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.index = None
        
    def build_index(self, documents: list):
        embeddings = self.encoder.encode(documents)
        dimension = embeddings.shape[1]
        
        # 归一化 + 余弦相似度
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
    def save_index(self, path: str):
        faiss.write_index(self.index, path)