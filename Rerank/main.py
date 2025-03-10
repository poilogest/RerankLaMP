from models.encoder import Encoder
from models.generator import SimpleGenerator
from retrieval.vector_indexer import VectorIndexer
from retrieval.vector_retriever import VectorRetriever
from models.base_rag import BaseRAG
from retrieval.bm25_indexer import Bm25Indexer
from retrieval.bm25_retriever import Bm25Retriever



if __name__ == "__main__":
    encoder = Encoder("sentence-transformers/all-mpnet-base-v2")
    vector_indexer = VectorIndexer(encoder)
    docs = ["Paris is the capital and largest city of France.", "France is a country.", "Python is an interpreted high-level programing language."]
    vector_indexer.build_index(docs)
    vector_retriever = VectorRetriever(vector_indexer, encoder)
    generator = SimpleGenerator()
    
    # 示例查询
    question = "Where is Paris?"
    
    vector_rag = BaseRAG(vector_retriever, generator)

    response = vector_rag(question, docs)

    print(response)

    bm25_indexer = Bm25Indexer()
    bm25_indexer.build_index(docs)
    bm25_retriever = Bm25Retriever(bm25_indexer)
    bm25_rag = BaseRAG(bm25_retriever, generator)
    response = bm25_rag(question, docs)

    print(response)
