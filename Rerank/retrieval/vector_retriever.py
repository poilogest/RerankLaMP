import numpy as np
from typing import List
import faiss

class VectorRetriever:
    def __init__(self, indexer, encoder):
        self.indexer = indexer
        self.encoder = encoder
        
    def retrieve(self, query: str, k: int=3) -> List[str]:
        query_embed = self.encoder.encode([query])
        faiss.normalize_L2(query_embed)
        scores, indices = self.indexer.index.search(query_embed, k)
        return indices[0]