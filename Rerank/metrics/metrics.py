# metrics.py
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from rouge_score import rouge_scorer
import numpy as np

class LaMPEvaluator:
    _METRIC_MAP = {
        "LaMP_1": {
            "type": "classification",
            "metrics": {
                "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred)
            }
        },
        "LaMP_2": {
            "type": "classification",
            "metrics": {
                "accuracy": accuracy_score,
                "f1_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro')
            }
        },
        "LaMP_3": {
            "type": "regression", 
            "metrics": {
                "mae": mean_absolute_error,
                "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
            }
        },
        "LaMP_4": {
            "type": "generation",
            "metrics": ["rouge1", "rougeL"]
        },
        "LaMP_5": {
            "type": "generation",
            "metrics": ["rouge1", "rougeL"]
        },
        "LaMP_6": {
            "type": "generation",
            "metrics": ["rouge1", "rougeL"]
        },
        "LaMP_7": {
            "type": "generation",
            "metrics": ["rouge1", "rougeL"]
        }
    }

    def __init__(self, task_id):
        
        if task_id not in self._METRIC_MAP:
            raise ValueError(f"Invalid task ID {task_id}. Supported: {list(self._METRIC_MAP.keys())}")
        
        self.task_id = task_id
        self.config = self._METRIC_MAP[task_id]
        
        if self.config["type"] == "generation":
            self.scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rougeL'], 
                use_stemmer=True
            )

    def compute(self, references, predictions):
        """
        统一计算入口
        :param references: 真实标签/文本列表
        :param predictions: 预测结果列表
        :return: 指标字典（自动保留4位小数）
        """
        if self.config["type"] == "generation":
            return self._compute_generation(references, predictions)
        else:
            return self._compute_classification(references, predictions)

    def _compute_classification(self, y_true, y_pred):
        results = {}
        for metric_name, metric_fn in self.config["metrics"].items():
            results[metric_name] = round(metric_fn(y_true, y_pred), 4)
        return results

    def _compute_generation(self, refs, preds):
        scores = []
        for ref, pred in zip(refs, preds):
            score = self.scorer.score(ref, pred)
            scores.append({k: v.fmeasure for k, v in score.items()})

        return {
            metric: round(np.mean([s[metric] for s in scores]), 4)
            for metric in self.config["metrics"]
        }

    @property
    def expected_input_type(self):
        """返回任务期望的输入数据类型"""
        return "text" if self.config["type"] == "generation" else "numeric"
    
if __name__ == "__main__":
    task_id = "LaMP_1"
    evaluator = LaMPEvaluator(task_id)
    references = [0, 1, 0, 1, 0]
    predictions = [0, 1, 1, 1, 0]
    results = evaluator.compute(references, predictions)
    print(results)