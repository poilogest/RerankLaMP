import math
import re
from collections import defaultdict

class Bm25Indexer:
    def __init__(self):
        # 核心数据结构
        self.inverted_index = defaultdict(dict)  # {term: {doc_id: tf}}
        self.doc_lengths = {}                    # {doc_id: document_length}
        self.doc_count = 0                       # 文档总数
        self.df = defaultdict(int)               # 文档频率 {term: df}
        self.idf = {}                            # 逆文档频率 {term: idf}
        self.avg_dl = 0                          # 平均文档长度

    def add_document(self, doc_id, text):
        """添加文档到索引"""
        tokens = re.findall(r'\w+', text.lower())  # 简单分词+小写化
        term_counts = defaultdict(int)
        
        # 计算词频
        for term in tokens:
            term_counts[term] += 1
        
        # 更新索引
        for term, tf in term_counts.items():
            self.inverted_index[term][doc_id] = tf
            self.df[term] += 1  # 文档频率只需计数一次
            
        # 记录文档长度
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_count += 1

    def build(self):
        """构建最终索引结构，计算统计量"""
        # 计算IDF (with smoothing)
        for term, df in self.df.items():
            self.idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        
        # 计算平均文档长度
        total_length = sum(self.doc_lengths.values())
        self.avg_dl = total_length / self.doc_count if self.doc_count > 0 else 0

    def build_index(self, documents):  # 提供两种调用方式

        for i in range(len(documents)):
            self.add_document(i, documents[i])
        self.build()

    @property
    def vocabulary(self):
        """返回词表"""
        return list(self.inverted_index.keys())