# retrieval/base_retriever.py
from abc import ABC, abstractmethod

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list:
        """检索抽象方法"""
        pass

    