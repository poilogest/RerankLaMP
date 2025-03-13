# scripts/run_bm25_baseline.py
import json
import os
from pathlib import Path
from typing import Dict, List
from ..retrieval.bm25_indexer import Bm25Indexer
from ..retrieval.bm25_retriever import Bm25Retriever
from ..models.base_rag import BaseRAG
from ..models.generator import SimpleGenerator
from ..evaluation.eval import LaMPEvaluator
from ..utils.config_loader import load_config

class LaMPDataLoader:
    """通用LaMP数据加载器"""
    def __init__(self, task_id: str, data_dir: str = "data"):
        self.task_id = task_id
        self.data_dir = Path(data_dir) / f"LaMP_{task_id[-1]}"
        
    def _load_json(self, filename: str) -> Dict:
        with open(self.data_dir / filename) as f:
            return json.load(f)
    
    def load_data(self, split: str) -> List[Dict]:
        """加载指定split的数据集"""
        questions = self._load_json(f"{split}_questions.json")
        outputs = self._load_json(f"{split}_outputs.json")
        
        # 对齐数据
        data_map = {item["id"]: item for item in outputs["golds"]}
        
        samples = []
        for q in questions:
            sample = {
                "id": q["id"],
                "input": q["input"],
                "profile": q["profile"],
                "output": data_map[q["id"]]["output"]
            }
            samples.append(sample)
        return samples

class BM25Pipeline:
    """BM25基准测试流水线"""
    def __init__(self, config):
        self.config = config
        self.generator = SimpleGenerator()
        self.retriever = None
        self.evaluator = LaMPEvaluator(None)  # 评估器独立使用
        
    def _build_user_index(self, profile: List[Dict], task_id: str) -> Bm25Retriever:
        """根据用户画像构建BM25索引"""
        indexer = Bm25Indexer()
        
        # 根据任务类型拼接文档内容
        for idx, item in enumerate(profile):
            if task_id == "LaMP1":
                text = f"{item['title']} {item['abstract']}"
            elif task_id in ["LaMP2", "LaMP4", "LaMP5"]:
                text = f"{item['title']} {item.get('text', '')}"
            elif task_id == "LaMP3":
                text = item['text']
            elif task_id == "LaMP6":
                text = item['text']  # 文件名作为文本
            elif task_id == "LaMP7":
                text = item['text']
            else:
                raise ValueError(f"Unknown task: {task_id}")
            
            indexer.add_document(doc_id=idx, text=text)
        
        indexer.build()
        return Bm25Retriever(indexer)
    
    def process_sample(self, sample: Dict, task_id: str) -> str:
        """处理单个样本"""
        # 1. 构建用户画像索引
        self.retriever = self._build_user_index(sample["profile"], task_id)
        
        # 2. 检索相关文档
        retrieved_docs = self.retriever.retrieve(
            query=sample["input"],
            k=self.config.retriever.top_k
        )
        
        # 3. 获取检索文本
        context = []
        for doc_id in retrieved_docs:
            profile_item = sample["profile"][doc_id]
            if task_id == "LaMP1":
                context.append(f"Title: {profile_item['title']}\nAbstract: {profile_item['abstract']}")
            elif task_id == "LaMP3":
                context.append(f"Review: {profile_item['text']}\nScore: {profile_item['score']}")
            else:
                context.append(profile_item.get('text', profile_item.get('title', '')))
        
        # 4. 生成预测
        if task_id in ["LaMP1", "LaMP2", "LaMP3"]:
            # 分类任务：简单基于检索结果的投票（实际项目应替换为分类模型）
            if task_id == "LaMP1":
                return "1" if len(context) > 0 else "0"
            scores = [item.get("score", 1) for item in sample["profile"]][:len(context)]
            return max(set(scores), key=scores.count) if scores else "0"
        else:
            # 生成任务
            return self.generator.generate(sample["input"], context)
    
    def evaluate_task(self, task_id: str, split: str = "test"):
        """评估指定任务"""
        loader = LaMPDataLoader(task_id)
        samples = loader.load_data(split)
        
        predictions = []
        for sample in samples:
            pred = self.process_sample(sample, task_id)
            predictions.append({
                "id": sample["id"],
                "prediction": pred,
                "reference": sample["output"]
            })
        
        # 转换为评估格式
        hypotheses = [p["prediction"] for p in predictions]
        references = [p["reference"] for p in predictions]
        
        # 根据任务类型选择评估指标
        if task_id in ["LaMP1", "LaMP2"]:
            y_true = [int(ref) for ref in references]
            y_pred = [int(pred) for pred in hypotheses]
            return self.evaluator._evaluate_classification(task_id, y_true, y_pred)
        elif task_id == "LaMP3":
            y_true = [float(ref) for ref in references]
            y_pred = [float(pred) for pred in hypotheses]
            return self.evaluator._evaluate_ordinal(task_id, y_true, y_pred)
        else:
            return self.evaluator._evaluate_generation(task_id, hypotheses, references)

def main():
    # 加载配置
    config = load_config(
        base_path="configs/base.yaml",
        override_paths=["configs/model_config.yaml"]
    )
    
    # 初始化流水线
    pipeline = BM25Pipeline(config)
    
    # 运行所有任务基准测试
    for task_id in [f"LaMP_{i}" for i in range(1, 8)]:
        print(f"\n{'='*30} Evaluating {task_id} {'='*30}")
        results = pipeline.evaluate_task(task_id)
        print(f"{task_id} Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()