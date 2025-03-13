# evaluation/eval.py
import torch
from tqdm import tqdm
from typing import Dict, Union
from metrics import (
    binary_accuracy,
    multi_class_metrics,
    ordinal_metrics
)
from metrics import rouge_scores

class LaMPEvaluator:
    def __init__(self, 
                 model: torch.nn.Module,
                 tokenizer: Union[object, None] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化评估器
        :param model: 待评估的模型
        :param tokenizer: 文本生成任务的tokenizer（可选）
        :param device: 计算设备
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.task_handlers = {
            "LaMP1": self._evaluate_classification,
            "LaMP2": self._evaluate_classification,
            "LaMP3": self._evaluate_ordinal,
            "LaMP4": self._evaluate_generation,
            "LaMP5": self._evaluate_generation,
            "LaMP6": self._evaluate_generation,
            "LaMP7": self._evaluate_generation
        }

    def evaluate(self, 
                 task_name: str, 
                 dataloader: torch.utils.data.DataLoader,
                 generation_kwargs: Dict = None) -> Dict:
        """
        统一评估入口
        :param task_name: 任务名称 LaMP1-LaMP7
        :param dataloader: 数据加载器
        :param generation_kwargs: 生成任务的参数配置
        :return: 评估指标字典
        """
        handler = self.task_handlers.get(task_name)
        if not handler:
            raise ValueError(f"Unsupported task: {task_name}. Available: {self.task_handlers.keys()}")

        return handler(task_name, dataloader, generation_kwargs or {})

    def _evaluate_classification(self, 
                                task_name: str,
                                dataloader: torch.utils.data.DataLoader,
                                _) -> Dict:
        """处理分类任务（LaMP1/2）"""
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                inputs = batch["input"].to(self.device)
                outputs = self.model(inputs)
                
                y_true.extend(batch["label"].cpu().tolist())
                y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

        # 根据任务类型选择指标
        if task_name == "LaMP1":
            return {"accuracy": binary_accuracy(y_true, y_pred)}
        return multi_class_metrics(y_true, y_pred)

    def _evaluate_ordinal(self,
                        task_name: str,
                        dataloader: torch.utils.data.DataLoader,
                        _) -> Dict:
        """处理序数回归任务（LaMP3）"""
        self.model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                inputs = batch["input"].to(self.device)
                outputs = self.model(inputs).squeeze()
                
                targets.extend(batch["label"].cpu().tolist())
                predictions.extend(outputs.cpu().tolist())

        return ordinal_metrics(targets, predictions)

    def _evaluate_generation(self,
                           task_name: str,
                           dataloader: torch.utils.data.DataLoader,
                           generation_kwargs: Dict) -> Dict:
        """处理生成任务（LaMP4-7）"""
        self.model.eval()
        hypotheses, references = [], []
        gen_args = {"max_length": 512, **generation_kwargs}

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
                inputs = self.tokenizer(
                    batch["input"], 
                    padding=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    **gen_args
                )
                decoded = self.tokenizer.batch_decode(
                    outputs, 
                    skip_special_tokens=True
                )
                
                hypotheses.extend(decoded)
                references.extend(batch["reference"])

        return rouge_scores(hypotheses, references)

# 使用示例
if __name__ == "__main__":
    # 初始化组件
    from transformers import AutoModel, AutoTokenizer
    
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    evaluator = LaMPEvaluator(model, tokenizer)

    # 假设已有DataLoader
    lamp1_loader = ...  # 分类任务数据加载器
    lamp4_loader = ...  # 生成任务数据加载器

    # 执行评估
    lamp1_results = evaluator.evaluate("LaMP1", lamp1_loader)
    lamp4_results = evaluator.evaluate("LaMP4", lamp4_loader, {"max_length": 128})
    
    print(f"LaMP1 Results: {lamp1_results}")
    print(f"LaMP4 Results: {lamp4_results}")