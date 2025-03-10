from torch import nn

class BaseRAG(nn.Module):
    def __init__(self, retriever, generator):
        super().__init__()
        self.retriever = retriever  # 例如BM25
        self.generator = generator  # 例如GPT-2

    def forward(self, question, docs):
        
        context_indices = self.retriever.retrieve(question)
        context = [docs[i] for i in context_indices]
        response = self.generator.generate(question, context)
        return response