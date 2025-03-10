# utils/gpt.py
import requests
import json
import yaml
from pathlib import Path
from typing import Dict, Any
from config_loader import load_config



class SimpleGPT:
    def __init__(self, config_path: str = "configs/gpt_api.yaml"):
        # 加载配置文件
        config = load_config(config_path)
        
        # 初始化API参数
        self.url = config.get("api_endpoint")
        self.api_key = config.get("api_key")
        
        # 设置请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    

    def generate(self, prompt: str, model: str = "DeepSeek-R1-671B", temperature: float = 0.7) -> str:
        """执行API调用"""
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=data,
                timeout=10
            )
            response.raise_for_status()  # 检查HTTP错误
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            return f"请求失败: {str(e)}"
        except (KeyError, json.JSONDecodeError) as e:
            return f"响应解析失败: {str(e)}"

if __name__ == "__main__":
    # 示例配置路径（根据实际位置调整）
    config_path = "configs/gpt_api.yaml"  # 假设配置文件在上级configs目录
    
    try:
        gpt = SimpleGPT(config_path)
        print("测试生成结果：")
        print(gpt.generate("This is a test."))
    except Exception as e:
        print(f"初始化失败: {str(e)}")