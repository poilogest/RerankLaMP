import random
from copy import deepcopy
from typing import List, Dict
from rank_bm25 import BM25Okapi
from prompts.utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_after_colon, extract_after_abstract, extract_after_description

from typing import List, Dict



def build_corpus_and_query(task_type: str, inp: str, profile: list, config: dict, use_date = False) -> tuple:
    # 获取配置参数
    corpus_fields = config['fields']
    query_extractor_name = config['query_extractor']
    
    try:
        # 使用globals动态获取提取函数
        query_extractor = globals()[query_extractor_name]
    except KeyError:
        raise ValueError(f"无效的查询提取器: {query_extractor_name}")

    # 构建语料
    corpus = []
    for p in profile:
        parts = []
        for field in corpus_fields:
            # 添加字段验证
            if field not in p:
                raise KeyError(f"字段 {field} 不存在于profile数据中")
            parts.append(str(p[field]))
            
        if use_date and 'date' in p:
            parts.append(f"date: {p['date']}")
            
        corpus.append(" ".join(parts))
    
    # 提取查询
    if task_type == "LaMP-1":
        extracted = query_extractor(inp)
        query = f'{extracted[1]} {extracted[2]}'
    else:
        query = query_extractor(inp)
    
    return corpus, query, [p['id'] for p in profile]


def random_select_profiles(source_data: List[Dict], k: int = 1, 
                          seed: int = None) -> List[Dict]:
    """
    随机选择profiles策略
    Args:
        source_data: 原始数据列表，每个元素需包含'profile'字段
        k: 选择数量
        seed: 随机种子
    """
    processed = deepcopy(source_data)
    if seed is not None:
        random.seed(seed)
    
    for data in processed:
        profiles = data['profile']
        data['profile'] = random.choices(profiles, k=k)
    return processed

def bm25_select_profiles(source_data: List[Dict], task, config, k: int = 1) -> List[Dict]:
    """
    基于BM25的相关性选择策略
    Args:
        source_data: 原始数据列表，每个元素需包含'input'和'profile'字段
        k: 选择数量
    """
    processed = deepcopy(source_data)
    
    for data in processed:
        profiles = data['profile']
        corpus, query, ids = build_corpus_and_query(task, data['input'], profiles, config)
        bm25 = BM25Okapi(corpus)
        
        # 计算相关性得分
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        
        # 获取Top-K结果
        top_indices = sorted(range(len(scores)),
                           key=lambda i: scores[i], reverse=True)[:k]
        selected = [profiles[i] for i in top_indices]
        
        # 处理结果不足的情况
        if len(selected) < k:
            selected += [profiles[-1]] * (k - len(selected))
            
        data['profile'] = selected
    
    return processed