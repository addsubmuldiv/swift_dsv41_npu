#!/usr/bin/env python3
"""
将本地修改通过 git push 推到远端，再由远程服务器拉取。

流程:
  本地: 读取 git remote url + 当前 branch → git push origin <branch>
  远程: 检查 remote url 是否一致
    一致 → git fetch + checkout <branch> + reset --hard origin/<branch>
    不一致/非 git 目录 → rm -rf + git clone + checkout <branch>
  可选: 拉取后直接跑 pytest

配置读写 scripts/.env_npu（与 run_npu_ut.py 共用）。.env_npu 只保存非敏感配置；
密码只能来自 NPU_PASSWORD 环境变量或 SSH agent/keychain。
"""

import argparse
import os
import shutil
import subprocess
import sys

import paramiko


def _load_dotenv(path):
    result = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                result[k.strip()] = v.strip().strip("'").strip('"')
    return result


# ── 本地 git 操作 ────────────────────────────────────────────────────────────


def _get_local_git_info():
    """返回 (remote_url, current_branch, repo_name)"""

    def _git(args):
        return subprocess.check_output(["git"] + args, text=True).strip()

    url = _git(["remote", "get-url", "origin"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    # 从 url 推断仓库名（取最后一段，去掉 .git）
    name = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return url, branch, name


def _push_local(branch: str):
    """推送当前分支到 origin"""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable was not found on PATH")
    subprocess.run([git, "push", "origin", branch], check=True)


# ── 远程 git 操作 ────────────────────────────────────────────────────────────

_REMOTE_CHECK = """
    cd {workdir} 2>/dev/null || true
    if [ -d .git ]; then
        remote=$(git remote get-url origin 2>/dev/null || echo "")
        if [ "$remote" = "{expected_url}" ]; then
            echo "MATCH"
        else
            echo "MISMATCH:$remote"
        fi
    else
        echo "NO_GIT"
    fi
"""

_REMOTE_SYNC_MATCH = """
    cd {workdir}
    git fetch origin
    git checkout {branch}
    git reset --hard origin/{branch}
    echo "SYNC_OK"
"""

_REMOTE_SYNC_CLONE = """
    cd $(dirname {workdir})
    rm -rf {repo_name}
    git clone {url}
    cd {repo_name}
    git checkout {branch}
    echo "CLONE_OK"
"""


def _run_remote(ssh_client, cmd: str):
    """执行远程命令，打印输出，返回是否成功"""
    stdin, stdout, stderr = ssh_client.exec_command(cmd)
    out_text = stdout.read().decode()
    err_text = stderr.read().decode()
    if out_text:
        print(out_text, end="")
    if err_text:
        print(err_text, end="")
    return out_text


def _sync_remote(ssh_client, workdir: str, url: str, branch: str, repo_name: str):
    """在远程执行核验 + 同步"""
    # 1) 核验
    check_cmd = _REMOTE_CHECK.format(workdir=workdir, expected_url=url)
    print(f"[REMOTE] 核验 {workdir} ...")
    result = _run_remote(ssh_client, check_cmd)

    # 2) 拉取或克隆
    if "MATCH" in result:
        print(f"[REMOTE] remote url 一致 → git pull ({branch})")
        cmd = _REMOTE_SYNC_MATCH.format(workdir=workdir, branch=branch)
    elif "MISMATCH" in result:
        mismatch_url = result.split("MISMATCH:")[-1].strip()
        print(f"[REMOTE] remote url 不一致 ({mismatch_url}) → 删除 + git clone {url}")
        cmd = _REMOTE_SYNC_CLONE.format(
            workdir=workdir, url=url, branch=branch, repo_name=repo_name
        )
    else:  # NO_GIT 或目录不存在
        print(f"[REMOTE] 非 git 目录或不存在 → git clone {url}")
        cmd = _REMOTE_SYNC_CLONE.format(
            workdir=workdir, url=url, branch=branch, repo_name=repo_name
        )

    _run_remote(ssh_client, cmd)


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    env_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv = _load_dotenv(os.path.join(env_dir, ".env_npu"))

    host = dotenv.get("NPU_HOST", os.getenv("NPU_HOST"))
    user = dotenv.get("NPU_USER", os.getenv("NPU_USER"))
    password = os.getenv("NPU_PASSWORD", "") or None
    workdir = dotenv.get("NPU_WORKDIR", os.getenv("NPU_WORKDIR"))
    conda_bin = dotenv.get("NPU_CONDA_BIN", os.getenv("NPU_CONDA_BIN"))
    env_script = dotenv.get("NPU_ENV_SCRIPT", os.getenv("NPU_ENV_SCRIPT"))
    atb_script = dotenv.get("NPU_ATB_SCRIPT", os.getenv("NPU_ATB_SCRIPT", ""))

    parser = argparse.ArgumentParser(description="git push + 远程拉取 + 可选跑测试")
    parser.add_argument("--host", default=host)
    parser.add_argument("--user", default=user)
    parser.add_argument("--workdir", default=workdir)
    parser.add_argument("--run", action="store_true", help="同步后跑全量 pytest")
    parser.add_argument(
        "pytest_args", nargs="*", default=[], help="传给 pytest 的参数（--run 时生效）"
    )

    args = parser.parse_args()

    if not all([args.host, args.user, args.workdir]):
        print("[ERROR] 缺少远程连接信息。先配好 --save-config 或 .env_npu 文件。")
        sys.exit(1)

    # ── 本地 push ──
    url, branch, repo_name = _get_local_git_info()
    print(f"[LOCAL] origin = {url}, branch = {branch}")
    _push_local(branch)
    print(f"[LOCAL] pushed {branch} → origin")

    # ── 远程 sync ──
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            args.host,
            username=args.user,
            password=password,
            timeout=30,
            look_for_keys=password is None,
            allow_agent=password is None,
        )
        _sync_remote(client, args.workdir, url, branch, repo_name)

        # ── 可选跑测试 ──
        if args.run:
            env_cmd = f"source {env_script}"
            if atb_script:
                env_cmd += f" && source {atb_script}"
            pytest_cmd = (
                " ".join(args.pytest_args)
                if args.pytest_args
                else "tests/pytorch/ -v --cov=transformer_engine"
            )
            cmd = f"{env_cmd} && export PATH={conda_bin}:$PATH && cd {args.workdir} && pytest {pytest_cmd}"
            print("[REMOTE] pytest ...")
            _run_remote(client, cmd)

    except paramiko.AuthenticationException:
        print("[ERROR] SSH 认证失败。")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
