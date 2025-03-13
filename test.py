import json
from pathlib import Path

def validate_id_matching(task_dir: Path):
    """验证同一索引位置的问题ID与输出ID是否一致"""
    try:
        # 读取问题文件
        with open(task_dir / "train_questions.json") as f:
            questions = json.load(f)  # 预期结构: List[Dict]
        
        # 读取输出文件
        with open(task_dir / "train_outputs.json") as f:
            outputs = json.load(f)    # 预期结构: Dict with "golds" list
    except FileNotFoundError as e:
        print(f"❌ 文件缺失: {e.filename}")
        return
    except json.JSONDecodeError:
        print(f"❌ JSON解析失败: {task_dir}")
        return

    # 验证数据结构
    if not isinstance(questions, list) or not all("id" in q for q in questions):
        print(f"❌ 问题文件结构异常: 应为List[Dict]且包含id字段")
        return
    if "golds" not in outputs or not all("id" in g for g in outputs["golds"]):
        print(f"❌ 输出文件结构异常: 缺少golds列表或id字段")
        return

    # 提取ID列表
    q_ids = [str(q["id"]) for q in questions]  # 统一转为字符串比较
    g_ids = [str(g["id"]) for g in outputs["golds"]]

    # 检查列表长度
    if len(q_ids) != len(g_ids):
        print(f"⚠️ 数据量不匹配: 问题数({len(q_ids)}) ≠ 输出数({len(g_ids)})")
        return

    # 索引匹配验证
    mismatches = []
    for idx, (q_id, g_id) in enumerate(zip(q_ids, g_ids)):
        if q_id != g_id:
            mismatches.append( (idx+1, q_id, g_id) )  # 索引从1开始计数

    # 输出结果
    if mismatches:
        print(f"🔴 发现 {len(mismatches)} 处ID不匹配（示例）：")
        for pos, q, g in mismatches[:3]:
            print(f"   第{pos}条: 问题ID={q} → 输出ID={g}")
        if len(mismatches) > 3:
            print(f"   ...（其他{len(mismatches)-3}处省略）")
    else:
        print(f"🟢 完全匹配（共{len(q_ids)}条数据）")

def main():
    base_path = Path("")  # 修改为实际路径
    
    for task_num in range(1, 8):
        task_dir = base_path / f"LaMP_{task_num}"
        print(f"\n=== 正在验证 {task_dir.name} ===")
        
        if task_dir.exists():
            validate_id_matching(task_dir)
        else:
            print(f"⏩ 目录不存在，已跳过")

if __name__ == "__main__":
    main()