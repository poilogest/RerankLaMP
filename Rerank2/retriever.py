import random
from copy import deepcopy
from typing import List, Dict
from rank_bm25 import BM25Okapi
from prompts.utils import extract_strings_between_quotes, extract_after_article, extract_after_review, extract_after_paper, add_string_after_title, extract_after_colon, extract_after_abstract, extract_after_description

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from tqdm.auto import tqdm
from functools import wraps
import time
import torch


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
        profiles_clusters = data['profile']
        selected = []
        cluster_count = len(profiles_clusters)
        base_k = max(k // cluster_count, 1)  # 基础分配数
        remainder = k % cluster_count        # 余数分配
        for i, profiles in enumerate(profiles_clusters):
            current_k = base_k + (1 if i < remainder else 0)
            selected += random.choices(profiles, k=current_k)
        data['profile'] = selected[:k]
    return processed

def bm25_select_profiles(source_data: List[Dict], task, config, k: int = 1) -> List[Dict]:
    """
    基于BM25的带中途补位策略的选择器
    Args:
        source_data: 原始数据列表，每个元素需包含'input'和'profile'字段
        k: 严格对齐的目标数量，中途自动插入None占位
    """
    processed = deepcopy(source_data)
    
    for data in processed:
        clusters = data['profile']
        selected = []
        cluster_num = len(clusters)
        
        # 空输入处理
        if cluster_num == 0:
            data['profile'] = [None]*k
            continue
        
        # 精确k值分配
        base_quota = k // cluster_num
        remainder = k % cluster_num
        
        for idx, cluster in enumerate(clusters):
            # 计算当前cluster的配额
            quota = base_quota + (1 if idx < remainder else 0)
            if quota <= 0:
                continue
            
            # 空cluster直接补位
            if not cluster:
                selected += [None]*quota
                continue
                
            try:
                # 语料构建
                corpus, query, _ = build_corpus_and_query(task, data['input'], cluster, config)
                
                # 空语料补位
                if not corpus:
                    selected += [None]*quota
                    continue
                
                # BM25处理
                bm25 = BM25Okapi(corpus)
                tokenized_query = query.split()
                scores = bm25.get_scores(tokenized_query)
                
                # 获取有效结果
                valid_num = min(quota, len(cluster))
                top_indices = sorted(range(len(scores)),
                                   key=lambda i: scores[i], reverse=True)[:valid_num]
                valid_results = [cluster[i] for i in top_indices]
                
                # 本cluster内补位
                valid_results += [None]*(quota - valid_num)
                selected.extend(valid_results)
                
            except Exception:
                # 异常情况完整补位
                selected += [None]*quota

        # 防御性截断（理论无需）
        data['profile'] = selected
    
    return processed


def k_means_cluster(source_data, task, config, k = 1):
    processed = deepcopy(source_data)
    for data in processed:
        profiles = data['profile']
        corpus, _, _ = build_corpus_and_query(task, data['input'], profiles, config)
        vectorizer = TfidfVectorizer()
        try:
            X = vectorizer.fit_transform(corpus)
        except ValueError:
            data['profile'] = profiles
            continue
        actual_clusters = min(k, X.shape[0]-1)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        clusters = [[] for _ in range(actual_clusters)]
        data["labels"] = {}
        for i, label in enumerate(labels):
            clusters[label].append(profiles[i])
            data["labels"][profiles[i]["id"]] = label
        data['profile'] = clusters
    return processed



def deep_cluster(source_data, task, config, model, tokenizer, k = 1):
    batch_size = 16
    processed = deepcopy(source_data)
    device = model.device
    # 主进度条：按data条目显示
    main_pbar = tqdm(
        processed,
        desc="处理数据条目",
        unit="data",
        dynamic_ncols=True
    )
    
    for data in main_pbar:
        profiles = data['profile']
        corpus, _, _ = build_corpus_and_query(task, data['input'], profiles, config)
        
        # 可选：在进度条右侧显示当前处理的data特征
        main_pbar.set_postfix({
            "profiles数": len(profiles),
        })
        
        all_labels = []
        # 内部处理保持静默（无进度条）
        for i in range(0, len(corpus), batch_size):
            batch_corpus = corpus[i:i+batch_size]
            
            inputs = tokenizer(
                batch_corpus,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            
            
            batch_logits = model.predict(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            
            batch_labels = batch_logits.to("cpu").numpy()
            
            all_labels.extend(batch_labels)
        
        # 聚类处理
        clusters = [[] for _ in range(k)]
        data["labels"] = {}
        for profile, label in zip(profiles, all_labels):
            cluster_idx = np.argmax(label)
            clusters[cluster_idx].append(profile)
            data["labels"][profile["id"]] = label.tolist()
        
        data['profile'] = clusters
        
        # 可选：更新进度条后缀为最新完成的数据ID
        if 'id' in data:
            main_pbar.set_postfix({"当前ID": data['id']})
    
    main_pbar.close()
    return processed


