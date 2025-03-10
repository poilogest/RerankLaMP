from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Encoder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)