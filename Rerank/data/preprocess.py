import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)

@dataclass
class LampSample:
    """统一数据结构类"""
    task: str
    sample_id: str
    input_text: str
    output: str
    profile: List[Dict[str, str]]
    split_type: str  # 'user' or 'time'

class LampPreprocessor:
    def __init__(self, raw_dir: str = "data/raw", output_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self._create_dirs()
        
        # 任务处理映射表
        self.task_processors = {
            "LaMP_1": self._process_lamp1,
            "LaMP_2": self._process_lamp2,
            #"LaMP_3": self._process_lamp3,
            #"LaMP_4": self._process_lamp4,
            #"LaMP_5": self._process_lamp5,
            #"LaMP_6": self._process_lamp6,
            #"LaMP_7": self._process_lamp7,
        }

    def _create_dirs(self):
        """创建输出目录结构"""
        (self.output_dir / "user_split").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "time_split").mkdir(parents=True, exist_ok=True)

    def process_all(self):
        """处理所有检测到的任务"""
        for task_dir in self.raw_dir.glob("LaMP_*"):
            task_name = task_dir.name
            if task_name in self.task_processors:
                logging.info(f"🔨 开始处理任务: {task_name}")
                self.task_processors[task_name](task_dir)
                logging.info(f"✅ 完成处理任务: {task_name}")
            else:
                logging.warning(f"⚠️ 发现未定义任务目录: {task_name}")

    def _load_task_data(self, task_dir: Path):
        """加载任务数据"""
        with open(task_dir / "train_questions.json") as f:
            questions = json.load(f)
        
        with open(task_dir / "train_outputs.json") as f:
            outputs = json.load(f)
        
        return questions, outputs

    def _process_lamp1(self, task_dir: Path):
        """处理学术引用识别任务"""
        questions, outputs = self._load_task_data(task_dir)
        output_map = {o["id"]: o["output"] for o in outputs["golds"]}
        
        samples = []
        for q in questions:
            sample = LampSample(
                task="LaMP_1",
                sample_id=q["id"],
                input_text=self._clean_input(q["input"], prefix="For an author who has written"),
                output=self._parse_output(output_map[q["id"]], task=1),
                profile=[self._format_profile(p, fields=["title", "abstract"]) for p in q["profile"]],
                split_type=self._detect_split_type(q["id"])
            )
            samples.append(sample)
        
        self._save_samples(samples, task_dir.name)

    def _process_lamp2(self, task_dir: Path):
        """处理电影分类任务"""
        questions, outputs = self._load_task_data(task_dir)
        output_map = {o["id"]: o["output"] for o in outputs["golds"]}
        
        samples = []
        for q in questions:
            sample = LampSample(
                task="LaMP_2",
                sample_id=q["id"],
                input_text=self._clean_input(q["input"], prefix="Which category"),
                output=output_map[q["id"]].lower(),
                profile=[self._format_profile(p, fields=["text", "category", "title"]) for p in q["profile"]],
                split_type=self._detect_split_type(q["id"])
            )
            samples.append(sample)
        
        self._save_samples(samples, task_dir.name)

    # 其他任务处理函数结构类似，根据具体数据结构调整

    def _clean_input(self, text: str, prefix: str = None) -> str:
        """清洗输入文本"""
        if prefix and text.startswith(prefix):
            return text[len(prefix):].strip().capitalize()
        return text.strip()

    def _parse_output(self, output: str, task: int) -> str:
        """解析不同任务的输出格式"""
        if task == 1:
            return output.strip("[]")
        return output.strip()

    def _format_profile(self, profile_item: Dict, fields: List[str]) -> Dict:
        """格式化用户画像条目"""
        return {field: profile_item.get(field, "")[:1000] for field in fields}  # 限制长度

    def _detect_split_type(self, sample_id: str) -> str:
        """根据样本ID检测分割类型（示例逻辑）"""
        return "user" if int(sample_id) % 2 == 0 else "time"

    def _save_samples(self, samples: List[LampSample], task_name: str):
        """保存处理后的样本"""
        split_types = {}
        for sample in samples:
            split_types.setdefault(sample.split_type, []).append(sample)
        
        for split_type, split_samples in split_types.items():
            output_path = self.output_dir / f"{split_type}_split" / f"{task_name}.jsonl"
            
            with open(output_path, "w") as f:
                for sample in split_samples:
                    f.write(json.dumps({
                        "task": sample.task,
                        "id": sample.sample_id,
                        "input": sample.input_text,
                        "output": sample.output,
                        "profile": sample.profile
                    }) + "\n")
            
            logging.info(f"💾 保存 {len(split_samples)} 个样本到 {output_path}")

if __name__ == "__main__":
    processor = LampPreprocessor()
    try:
        processor.process_all()
        logging.info("🎉 所有数据处理完成！")
    except Exception as e:
        logging.error(f"❌ 处理失败: {str(e)}")
        raise