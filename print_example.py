import json
import os

def get_nested_data(data, keys):
    """递归获取嵌套数据"""
    if not keys or not data:
        return data
    current_key = keys[0]
    
    if isinstance(current_key, int):
        if isinstance(data, list) and len(data) > current_key:
            return get_nested_data(data[current_key], keys[1:])
    elif current_key in data:
        return get_nested_data(data[current_key], keys[1:])
    return None

def get_json_data(path, data_path=None):
    """
    从JSON文件获取特定路径的数据
    data_path格式示例：['profile', 0] 表示取profile列表的第一个元素
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
            if data_path:
                result = get_nested_data(data, data_path)
                return result if result is not None else "N/A"
            
            return data[0] if isinstance(data, list) else data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return "N/A"

def generate_markdown_examples():
    """生成Markdown格式示例"""
    md_content = ["# LaMP 数据集示例\n"]
    
    for dataset_num in range(1, 8):
        base_dir = f"LaMP_{dataset_num}"
        question_path = os.path.join(base_dir, "train_questions.json")
        output_path = os.path.join(base_dir, "train_outputs.json")
        
        # 获取数据
        question = get_json_data(question_path, data_path=[0, 'input'])
        profiles = get_json_data(question_path, data_path=[0, 'profile'])
        
        # 处理profile信息
        profile_count = len(profiles) if isinstance(profiles, list) else 0
        first_profile = profiles[0] if (isinstance(profiles, list) and profile_count > 0) else "N/A"
        
        gold_output = get_json_data(output_path, data_path=['golds', 0, 'output'])

        # 构建Markdown内容
        section = f"""
## LaMP-{dataset_num}

- ​**Profile总数**: {profile_count}
- ​**首个Profile**: `{first_profile}`
- ​**问题**: `{question}`
- ​**输出**: `{gold_output}`

---
"""
        md_content.append(section)
    
    return "\n".join(md_content)

if __name__ == "__main__":
    output_file = "LaMP_examples.md"
    md_code = generate_markdown_examples()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_code)
    
    print(f"Markdown文件已生成至：{os.path.abspath(output_file)}")
    print("建议使用支持Markdown的编辑器（如VS Code、Typora）查看")