import json
import os
from collections import defaultdict

def analyze_structure(data, path="", result=None):
    """递归分析JSON结构"""
    if result is None:
        result = defaultdict(lambda: {'types': set(), 'examples': set()})
    
    if isinstance(data, dict):
        result[path]['types'].add('object')
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            analyze_structure(v, new_path, result)
            # 记录示例值（最多3个不同的值）
            if len(result[new_path]['examples']) < 3:
                example = str(v)[:50]  # 截断长文本
                result[new_path]['examples'].add(example)
    elif isinstance(data, list):
        result[path]['types'].add('array')
        if data:
            analyze_structure(data[0], f"{path}[]", result)
    else:
        result[path]['types'].add(type(data).__name__)
    
    return result

def get_structure_summary(filepath):
    """获取文件结构摘要"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # 如果是数组，取第一个元素分析
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
        else:
            sample = data
        
        structure = analyze_structure(sample)
        return structure
    except Exception as e:
        print(f"解析 {filepath} 失败: {e}")
        return {}

def print_structure(structure):
    """打印结构信息"""
    for path in sorted(structure.keys()):
        info = structure[path]
        types = ', '.join(info['types'])
        examples = ', '.join(info['examples']) if info['examples'] else '-'
        print(f"{path.ljust(40)} | {types.ljust(20)} | 示例: {examples}")

def generate_structure_reports():
    """生成所有数据集的结构报告"""
    for dataset_num in range(1, 8):
        base_dir = f"LaMP_{dataset_num}"
        print(f"\n{'='*40}")
        print(f"分析数据集: {base_dir}")
        print(f"{'='*40}")
        
        # 分析问题文件
        q_path = os.path.join(base_dir, "train_questions.json")
        if os.path.exists(q_path):
            print(f"\n问题文件结构 ({q_path}):")
            q_structure = get_structure_summary(q_path)
            print_structure(q_structure)
        
        # 分析输出文件
        o_path = os.path.join(base_dir, "train_outputs.json")
        if os.path.exists(o_path):
            print(f"\n输出文件结构 ({o_path}):")
            o_structure = get_structure_summary(o_path)
            print_structure(o_structure)

if __name__ == "__main__":
    generate_structure_reports()
    print("\n结构报告生成完成！建议将输出重定向到文件：")
    print("  python script.py > structure_report.txt")