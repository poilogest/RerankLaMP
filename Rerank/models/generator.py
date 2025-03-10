# models/generator.py
from transformers import pipeline

class SimpleGenerator:
    def __init__(self):
        """初始化生成模型"""
        self.generator = pipeline(
            'text2text-generation',
            model='google/flan-t5-small'
        )
    
    def generate(self, question: str, context: list) -> str:
        """基础生成方法"""
        input_text = f"question:{question} context:{' '.join(context)}"
        result = self.generator(input_text, max_length=200)
        return result[0]['generated_text']

# 使用示例
if __name__ == "__main__":
    generator = SimpleGenerator()
    
    test_question = "where is Paris?"
    test_context = ["Paris is the capital and largest city of France."]
    
    print(generator.generate(test_question, test_context))  # 输出：巴黎