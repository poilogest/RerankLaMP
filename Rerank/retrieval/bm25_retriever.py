class Bm25Retriever:
    def __init__(self, indexer):
        self.indexer = indexer
        self.k1 = 1.5  # BM25调校参数，控制词频饱和度
        self.b = 0.75   # BM25长度归一化参数
    
    def retrieve(self, query, k=3):
        scores = {}  # 存储文档得分 {doc_id: total_score}
        terms = query.split()  # 简单空格分词
        
        # 获取平均文档长度
        avg_dl = self.indexer.avg_dl
        
        for term in terms:
            # 跳过未索引词项
            if term not in self.indexer.inverted_index:
                continue
            
            # 获取词项的IDF和倒排列表
            idf = self.indexer.idf.get(term, 0)
            postings = self.indexer.inverted_index[term]
            
            for doc_id, tf in postings.items():
                dl = self.indexer.doc_lengths[doc_id]  # 文档长度
                
                # 计算BM25单项得分
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (dl / avg_dl))
                term_score = idf * numerator / denominator
                
                # 累加词项得分到文档总分
                scores[doc_id] = scores.get(doc_id, 0) + term_score
        
        # 按得分降序排序，返回前k个文档ID
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:k]]