#!/usr/bin/env python3
"""
SSH to NPU machine, run pytest with coverage, collect results.

所有环境配置通过以下方式读取（优先级从高到低）：
  1. 命令行参数（非敏感项）
  2. 环境变量
  3. 当前目录下的 .env_npu 文件（仅非敏感 key=value 配置）
  4. 对于 SSH 认证，如果以上都未设置密码，尝试 ~/.ssh/id_* 密钥

不硬编码任何主机/用户/密码/路径。不要把密码写入 .env_npu；
敏感信息仅通过 NPU_PASSWORD 环境变量或 SSH agent/keychain 提供，
不要通过命令行参数传递密码。

用法:
  python run_npu_ut.py                              # 跑全量
  python run_npu_ut.py tests/pytorch/test_xxx.py     # 单文件
  python run_npu_ut.py -k test_sanity_linear          # pytest -k 过滤
  python run_npu_ut.py --install                      # 仅装依赖
  python run_npu_ut.py --save-config                  # 把当前参数写入 .env_npu
"""

import argparse
import os
import sys
import paramiko


def _load_dotenv(path: str) -> dict:
    """加载 .env_npu 文件"""
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


def _resolve_ssh_password(user: str, host: str, password: str | None) -> str | None:
    """密码未设置时尝试密钥认证；paramiko 会自行尝试 ~/.ssh/id_*"""
    if password:
        return password
    return None  # paramiko 默认走密钥


def main():
    # 1) 加载 .env_npu（最低优先级）
    dotenv = _load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env_npu"))

    # 2) 命令行参数
    parser = argparse.ArgumentParser(
        description="SSH → pytest on NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
环境变量 (备选):
  NPU_HOST       SSH 主机地址
  NPU_USER       SSH 用户名
  NPU_PASSWORD   SSH 密码 (敏感；不要写入 .env_npu，留空则用 ~/.ssh/id_* 密钥)
  NPU_WORKDIR    TENPU 仓库根目录 (如 /home/xxx/TransformerEngineNPU)
  NPU_CONDA_BIN  conda env 的 bin 目录 (如 /home/miniconda3/envs/xxx/bin)
  NPU_ENV_SCRIPT CANN set_env.sh 路径 (如 /home/cann/CANN/.../set_env.sh)
  NPU_ATB_SCRIPT ATB set_env.sh 路径 (可选)

示例:
  首次使用:
    设置 NPU_HOST/NPU_USER/NPU_WORKDIR/NPU_CONDA_BIN/NPU_ENV_SCRIPT。
    如需密码认证，在当前 shell 设置 NPU_PASSWORD，不要写入文件。
    python run_npu_ut.py --save-config

  日常使用:
    python run_npu_ut.py -k test_sanity_linear
        """,
    )
    parser.add_argument("--host", default=os.getenv("NPU_HOST", dotenv.get("NPU_HOST")))
    parser.add_argument("--user", default=os.getenv("NPU_USER", dotenv.get("NPU_USER")))
    parser.add_argument(
        "--workdir",
        default=os.getenv("NPU_WORKDIR", dotenv.get("NPU_WORKDIR")),
    )
    parser.add_argument(
        "--conda-bin",
        default=os.getenv("NPU_CONDA_BIN", dotenv.get("NPU_CONDA_BIN")),
    )
    parser.add_argument(
        "--env-script",
        default=os.getenv("NPU_ENV_SCRIPT", dotenv.get("NPU_ENV_SCRIPT")),
    )
    parser.add_argument(
        "--atb-script",
        default=os.getenv("NPU_ATB_SCRIPT", dotenv.get("NPU_ATB_SCRIPT", "")),
    )
    parser.add_argument("--install", action="store_true", help="仅安装 pytest-cov 依赖")
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="将当前非敏感参数保存到脚本同目录下的 .env_npu 文件",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        default=[],
        help="传给 pytest 的参数 (tests/pytorch/ -v --cov=transformer_engine 为默认值)",
    )

    args = parser.parse_args()

    # 3) 保存配置
    if args.save_config:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env_npu")
        lines = []
        for key in (
            "NPU_HOST",
            "NPU_USER",
            "NPU_WORKDIR",
            "NPU_CONDA_BIN",
            "NPU_ENV_SCRIPT",
            "NPU_ATB_SCRIPT",
        ):
            val = getattr(args, key.lower().removeprefix("npu_"), None)
            if val:
                lines.append(f'{key}="{val}"')
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[CONFIG] 已写入 {env_path}")
        if not any(vars(args).get(k) for k in ("pytest_args",) if k != "save_config"):
            return  # --save-config 单独使用时不需要执行测试

    # 4) 校验必要参数
    missing = []
    for name, value in [
        ("--host", args.host),
        ("--user", args.user),
        ("--workdir", args.workdir),
        ("--conda-bin", args.conda_bin),
        ("--env-script", args.env_script),
    ]:
        if not value:
            missing.append(name)
    if missing:
        print(f"[ERROR] 缺少必要参数: {', '.join(missing)}")
        print("  用 --save-config + 完整参数首次写入配置，或设置对应的环境变量。")
        sys.exit(1)

    # 5) 组装远程命令
    password = os.getenv("NPU_PASSWORD", "") or None

    cmd_parts = [f"source {args.env_script}"]
    if args.atb_script:
        cmd_parts.append(f"source {args.atb_script}")
    cmd_parts.append(f"export PATH={args.conda_bin}:$PATH")

    if args.install:
        cmd_parts.append("pip install pytest-cov")
    else:
        pytest_args = (
            " ".join(args.pytest_args)
            if args.pytest_args
            else "tests/pytorch/ -v --cov=transformer_engine"
        )
        cmd_parts.append(f"cd {args.workdir}")
        cmd_parts.append(f"pytest {pytest_args}")

    remote_cmd = " && ".join(cmd_parts)

    print(f"[CONNECT] {args.user}@{args.host}")
    print(f"[CMD] {remote_cmd}")
    print("=" * 80)

    # 6) SSH 执行
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
        stdin, stdout, stderr = client.exec_command(remote_cmd)

        for line in iter(stdout.readline, ""):
            print(line, end="")
        for line in iter(stderr.readline, ""):
            print(line, end="")

        exit_code = stdout.channel.recv_exit_status()
        print(f"\n{'=' * 80}")
        print(f"[EXIT: {exit_code}]")
        sys.exit(exit_code)
    except paramiko.AuthenticationException:
        print(
            "[ERROR] SSH 认证失败。请检查 SSH agent/keychain，或通过 NPU_PASSWORD 环境变量设置密码。"
        )
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
