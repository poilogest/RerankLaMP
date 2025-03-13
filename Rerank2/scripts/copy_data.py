import shutil
from pathlib import Path
import sys

def copy_LaMP_datasets(
    root_dir: Path = Path(__file__).parent.parent.parent,  # 自动定位到项目根目录
    lamp_versions: list = [1,2,3,4,5,6,7],
    skip_existing: bool = True
):
    """智能复制LaMP数据集到Rerank2/data"""
    # 路径定义
    src_base = root_dir  # LaMP_1~7直接在根目录
    dest_dir = root_dir / "Rerank2" / "data"
    
    # 创建目标目录
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 项目根目录: {root_dir}")
    print(f"🎯 目标位置: {dest_dir}\n")

    # 复制每个LaMP数据集
    for ver in lamp_versions:
        src = src_base / f"LaMP_{ver}"
        dest = dest_dir / f"LaMP_{ver}"
        
        # 验证源目录
        if not src.exists():
            print(f"⛔ 缺失: LaMP_{ver} (跳过)")
            continue
            
        # 处理已存在的目标目录
        if dest.exists():
            if skip_existing:
                print(f"⏭️ 已存在: LaMP_{ver} (跳过)")
                continue
            else:
                shutil.rmtree(dest)
                print(f"♻️ 已清除旧数据: LaMP_{ver}")

        # 执行复制
        try:
            shutil.copytree(src, dest)
            file_count = sum(1 for _ in src.glob("**/*") if _.is_file())
            print(f"✅ 成功复制: LaMP_{ver} ({file_count}个文件)")
        except Exception as e:
            print(f"❌ 复制失败 LaMP_{ver}: {str(e)}")

if __name__ == "__main__":
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="复制LaMP数据集")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在数据")
    parser.add_argument("--versions", type=int, nargs='+', default=[1,2,3,4,5,6,7], 
                      help="指定要复制的版本号，例如 --versions 1 3 5")
    args = parser.parse_args()

    # 执行复制
    copy_LaMP_datasets(
        lamp_versions=args.versions,
        skip_existing=not args.force
    )