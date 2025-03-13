import os
import paramiko
from scp import SCPClient, SCPException

# 配置参数
SSH_HOST = "10.120.18.240"
SSH_PORT = 6988
SSH_USER = "yyan269-Rerank"
SSH_KEY_PATH = None
REMOTE_BASE_PATH = "/hpc2hdd/home/yyan269/ReLaMP/RerankLaMP"
LOCAL_BASE_DIR = "./"

def create_ssh_client():
    """创建SSH连接并启用压缩"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        password = input(f"请输入 {SSH_USER}@{SSH_HOST} 的SSH密码: ")
        # 手动创建Transport并启用压缩
        transport = paramiko.Transport((SSH_HOST, SSH_PORT))
        transport.connect(username=SSH_USER, password=password)
        transport.use_compression(True)  # 关键修复：在Transport层启用压缩
        return paramiko.SSHClient.from_transport(transport)
    except Exception as e:
        print(f"SSH连接失败: {e}")
        return None

def transfer_directory(scp, local_dir, remote_dir):
    """递归传输目录（保持不变）"""
    try:
        ssh = scp.transport
        ssh.exec_command(f"mkdir -p {remote_dir}")
        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            if os.path.isfile(local_path):
                scp.put(local_path, f"{remote_dir}/{item}")
                print(f"[OK] 已传输: {os.path.basename(local_path)}")
            elif os.path.isdir(local_path):
                transfer_directory(scp, local_path, f"{remote_dir}/{item}")
    except SCPException as e:
        print(f"[ERROR] 传输失败: {e}")

def main():
    ssh_client = create_ssh_client()
    if not ssh_client:
        return

    # 移除了 compress=True 参数
    scp = SCPClient(ssh_client.get_transport())

    for task_id in range(1, 8):
        local_dir = os.path.join(LOCAL_BASE_DIR, f"LaMP_{task_id}")
        remote_dir = os.path.join(REMOTE_BASE_PATH, f"LaMP_{task_id}")
        if not os.path.exists(local_dir):
            print(f"[WARNING] 本地目录不存在: {local_dir}")
            continue
        print(f"\n======= 传输 LaMP_{task_id} =======")
        transfer_directory(scp, local_dir, remote_dir)

    scp.close()
    ssh_client.close()
    print("\n✅ 所有文件传输完成！")

if __name__ == "__main__":
    main()