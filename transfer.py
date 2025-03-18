import os
import paramiko
import getpass
from scp import SCPClient, SCPException

# 配置参数
SSH_HOST = "10.120.18.240"
SSH_PORT = 6988
SSH_USER = "yyan269-Rerank"
REMOTE_BASE_PATH = "/hpc2hdd/home/yyan269/ReLaMP/RerankLaMP"
LOCAL_BASE_DIR = "./"

def create_ssh_client():
    """创建SSH连接并启用压缩"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        password = getpass.getpass(f"请输入 {SSH_USER}@{SSH_HOST} 的SSH密码: ")
        client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=password, compress=True)
        return client
    except Exception as e:
        print(f"SSH连接失败: {e}")
        return None

def transfer_directory(ssh_client, scp, local_dir, remote_dir):
    """使用 SCP 递归传输整个目录"""
    try:
        # **改用 ssh_client.exec_command() 确保远程目录存在**
        stdin, stdout, stderr = ssh_client.exec_command(f"mkdir -p {remote_dir}")
        stderr_output = stderr.read().decode()
        if stderr_output:
            print(f"[WARNING] 远程目录创建警告: {stderr_output}")

        scp.put(local_dir, remote_dir, recursive=True)
        print(f"[OK] 目录 {os.path.basename(local_dir)} 传输完成")
    except SCPException as e:
        print(f"[ERROR] 目录 {os.path.basename(local_dir)} 传输失败: {e}")

def main():
    ssh_client = create_ssh_client()
    if not ssh_client:
        return

    with SCPClient(ssh_client.get_transport()) as scp:
        for task_id in range(1, 8):
            local_dir = os.path.join(LOCAL_BASE_DIR, f"LaMP-{task_id}")
            remote_dir = os.path.join(REMOTE_BASE_PATH, f"LaMP-{task_id}")
            if not os.path.exists(local_dir):
                print(f"[WARNING] 本地目录不存在: {local_dir}")
                continue
            print(f"\n======= 传输 LaMP-{task_id} =======")
            transfer_directory(ssh_client, scp, local_dir, remote_dir)

    ssh_client.close()
    print("\n✅ 所有文件传输完成！")

if __name__ == "__main__":
    main()
