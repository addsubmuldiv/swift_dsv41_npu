# 操作手册（手动执行 / 重验 / 推送）

> 本文件承载 SKILL.md 中**条件触发**的长命令。SKILL.md 各阶段标注「详见本文件」时才需读取。
> 变量约定：`<skill_dir>` = SKILL.md 所在目录；`<workspace>` = `/home/ma_daily_ci/workspace`。

## 手动执行注意事项

`nightly_orchestrator.py` 自动处理 conda 环境激活和 PYTHONPATH 设置。但**手动逐个执行 pipeline 脚本**时，以下脚本需要额外环境准备：

| 脚本 | 需要 conda | 需要 PYTHONPATH | 原因 |
|------|-----------|----------------|------|
| `nightly_envcheck.py` | ❌ | ❌ | 纯 ps / ss / npu-smi 检测 |
| `nightly_fetch.py` | ❌ | ❌ | 仅 git 操作 |
| `nightly_setup.py` | ✅ | ✅ | `import megatron_adaptor` 冒烟测试触发 `import megatron` |
| `nightly_sigcheck.py` | ✅ | ✅ | `inspect.signature` 需加载 Megatron 函数 |
| `nightly_launch.py` | ✅ | ✅ | 子进程继承当前环境运行训练脚本 |
| `nightly_verify.py` | ❌ | ❌ | 纯 JSON 日志分析 |
| `fetch_mrs.py` | ❌ | ❌ | 纯 git 操作 |
| `nightly_report.py` | ❌ | ❌ | 纯 JSON 聚合 |

**通用手动执行模板**（需要 conda + PYTHONPATH 的脚本）：

```bash
source <skill_dir>/scripts/setup_env.sh <conda_env>
export PYTHONPATH=<workspace>/Megatron-LM:$PYTHONPATH
python3 scripts/<script_name>.py <args...>
```

⚠️ **注意**：省略 PYTHONPATH 时 `nightly_setup.py` 冒烟测试失败的特征是 `ModuleNotFoundError: No module named 'megatron'`，不代表环境有问题——只是缺少 PYTHONPATH。

## 各步骤命令

### ③ LAUNCH 首步 ENVCHECK（NPU 占用预检）

```bash
# 手动预检（无需 conda/PYTHONPATH）：
#   exit 0 = NPU free；exit 2 = NPU 被其他训练任务占用（occupied）
# occupied 时 envcheck_report.json 给出占用者 user + 端口，供人工确认后再决定是否跳过/协调
python3 scripts/nightly_envcheck.py --output <state_dir>/envcheck_report.json
```

判定信号：HCCL NPU socket 端口（默认 16666，遵循 `HCCL_NPU_SOCKET_PORT_RANGE`）已被 `ss -ltn` 绑定；或存在非 `ma_daily_ci`/root 属主的训练进程（训练入口名级别：`torchrun`/`torch.distributed.run`/`pretrain_gpt`/`trainer.py`/`mindspeed`/`run_pretraining`/`megatron/pretraining`）。⚠️ 不要用裸 `megatron`/`fsdp`/`pretrain` 子串——skill 路径 `megatron-adaptor-sync` 会让 root 包装进程误报。npu-smi 摘要仅作上下文不单独判定。

编排器行为：预检在 `nightly_launch.py` 第一步执行，occupied → 写 `env_busy=true` 的 `training_log.json` 返回 0 → 该分支 SKIP（`overall="skip"`，exit 2 当全部分支 skip；不诊断、不建 Issue/PR）。

### ① FETCH + SETUP

```bash
# ①a FETCH — 纯 git 操作，无需 conda
python3 scripts/nightly_fetch.py --work-dir $WORK_DIR --last-good $LAST_GOOD \
    --megatron-target origin/main --ma-target origin/main --te-target origin/main \
    --output $STATE/env_report.json

# ①b SETUP — 需要 conda + PYTHONPATH（冒烟测试 import megatron_adaptor → import megatron）
source <skill_dir>/scripts/setup_env.sh ma_agent_py312
export PYTHONPATH=$WORK_DIR/Megatron-LM:$PYTHONPATH
python3 scripts/nightly_setup.py --env-report $STATE/env_report.json --output $STATE/setup_report.json
# 失败则终止：setup_report.status != "ok"
```

### ② SIGCHECK

```bash
python3 scripts/nightly_sigcheck.py --adaptor-path $ADAPTOR_CLONE --env-report $STATE/env_report.json --output $STATE/sigcheck_report.json
# sigcheck_report.overall == "fail" → 跳至 ⑤ FETCH_MRS
```

### ③ LAUNCH

```bash
source <skill_dir>/scripts/setup_env.sh ma_agent_py312
export PYTHONPATH=$WORK_DIR/Megatron-LM:$PYTHONPATH
python3 scripts/nightly_launch.py --adaptor-path $ADAPTOR_CLONE --branch $BRANCH --output $STATE/training_log.json
```

### ⑤ FETCH_MRS

```bash
python3 scripts/fetch_mrs.py --repo-path $WORK_DIR/Megatron-LM \
    --last-good $LAST_GOOD --target $TARGET \
    --adaptor-path $ADAPTOR_CLONE --output $STATE/diff_report.json
```

### ⑧ REPORT

```bash
python3 scripts/nightly_report.py \
    --env-report $STATE/env_report.json --setup-report $STATE/setup_report.json \
    --training-log $STATE/training_log.json --verify-report $STATE/verify_report.json \
    --eval-set scripts/nightly_eval_set.json --output $STATE/final_report.json
```

## 重验命令（⑦ 修复循环内）

**自包含**——自行确保三仓库在正确分支，不依赖共享 clone 的当前状态。

> ⚠️ **关键**：本阶段所有分支可能停在任意分支上，需要以干净的分支状态开始。重验命令**必须先 checkout 到正确分支再执行**，否则会在错误的代码上验证，造成误判。

```bash
# ① 确保三仓库在正确分支（不依赖共享 clone 当前状态）
cd <workspace>/Megatron-LM && \
  git fetch origin && \
  git checkout --force origin/<megatron_branch> && \
  git reset --hard HEAD && git clean -fd && \
cd <workspace>/MegatronAdaptor && \
  git fetch origin && \
  git checkout --force origin/<ma_branch> && \
cd <workspace>/TransformerEngineNPU && \
  git fetch origin && \
  git checkout --force origin/<te_branch> && \
# ② 注入 adaptor import + 创建 test 目录软链接
cd <workspace>/Megatron-LM && \
  grep -q "import megatron_adaptor" pretrain_gpt.py || \
  sed -i '/^import torch$/a import megatron_adaptor' pretrain_gpt.py && \
  # test 脚本在 MA repo，但需要以 Megatron-LM 为 CWD 执行（pretrain_gpt.py 路径）
  # 同时脚本内 `source test/system_tests/env_npu.sh` 是相对路径，需要软链接
  test -L test || ln -sf <workspace>/MegatronAdaptor/test test && \
# ③ 确认分支正确（防御性校验）
echo "=== Verifying on Megatron-LM branch: $(git branch --show-current) commit: $(git rev-parse --short HEAD) ===" && \
# ④ 激活环境 + 运行测试
source <skill_dir>/scripts/setup_env.sh <conda_env> && \
export PYTHONPATH=<workspace>/Megatron-LM:$PYTHONPATH && \
cd <workspace>/Megatron-LM && \
bash -e test/system_tests/feature_tests/<test_name>.sh \
  > <workspace>/output/<YYYY-MM>/<MM-DD>/auto_fix/<branch>/<test>/stdout.log \
  2> <workspace>/output/<YYYY-MM>/<MM-DD>/auto_fix/<branch>/<test>/stderr.log
```

参数说明：

| 参数 | 含义 | 来源 |
|------|------|------|
| `<workspace>` | `/home/ma_daily_ci/workspace` | 固定路径 |
| `<skill_dir>` | skill 目录 | `SKILL.md` 所在目录 |
| `<megatron_branch>` | 当前修复针对的 MLM 分支 | `env_report.json` → `megatron.branch` |
| `<ma_branch>` | 对应的 MA 分支 | `env_report.json` → `adaptor.branch` |
| `<te_branch>` | TE-NPU 分支 | `env_report.json` → `transformer_engine_npu.branch` |
| `<conda_env>` | Conda 环境名 | `scripts/branch_matrix.json` 分支配套表 |
| `<test_name>` | 失败的测试用例名 | `verify_report.json` → `failure_details` |
| `<date>` | 构建日期 YYYY-MM-DD | `env_report.json` → `fetch_time` |
| `<branch>` | 分支目录名 | megatron_branch（如 `main`、`core_r0.17.0`） |

## 回归检查

全部修复完成后，从当前分支的 `test/system_tests/test_cases.txt` 随机选 2-3 个原本通过的用例重跑。**重跑前同样需要 checkout 到正确分支**（使用同上重验命令的步骤①②③）。有回归则撤销相关修改。

## 提交推送（⑦ 每分支仅一个干净 commit，force push 个人仓）

敏感信息（remote URL、API token、author）从 `scripts/.push_config.json` 读取，不硬编码。分支配套 → `scripts/branch_matrix.json`。

```bash
# 从 config 读取
PUSH_REMOTE=$(python3 -c "import json; print(json.load(open('scripts/.push_config.json'))['repo']['remote_url'])")
AUTHOR_NAME=$(python3 -c "import json; print(json.load(open('scripts/.push_config.json'))['repo']['author_name'])")
AUTHOR_EMAIL=$(python3 -c "import json; print(json.load(open('scripts/.push_config.json'))['repo']['author_email'])")
# credential helper：token 由它从 0600 的 .push_config.json 读取，不进 argv/不落盘 config
CRED_HELPER="$(dirname "$(readlink -f scripts/.push_config.json)")/git-credential-helper.sh"

cd /home/ma_daily_ci/workspace/MegatronAdaptor
# remote URL 保持干净（set-url 覆盖可能残留的 tokenized URL）
git remote set-url <git_remote> "$PUSH_REMOTE" 2>/dev/null || git remote add <git_remote> "$PUSH_REMOTE"

# 0. 确保 pre-commit hooks 已安装（首次执行一次即可，后续幂等）
pre-commit install
test -f ./gitleaks || cp /home/ma_daily_ci/workspace/gitleaks ./gitleaks
chmod +x ./gitleaks

# ① 从上游基线干净重建分支
git fetch origin <megatron_branch>
git checkout -B daily_fix_<megatron_branch> origin/<megatron_branch>

# ② 修改文件 → pre-commit 校验（先 auto-fix）→ 再 commit
# ⚠️ 必须先跑 pre-commit 再 commit：pre-commit run --files 会触发 ruff --fix
# 自动修复工作区文件，如果先 commit 再跑，commit 里的代码仍是修前版本
git add <修复文件1> <修复文件2> ...
pre-commit run --files <修复文件1> <修复文件2> ... 2>&1
# 如果 pre-commit 修改了文件，重新 add
git diff --stat --exit-code || git add -u
# 现在 commit — pre-commit hook 也会自动触发但文件已修复 → 全部 PASS
git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" commit \
  -m "fix(agent): <描述> [<测试名>] [branch: <megatron_branch>]"

# ③ 强制推送单 commit（认证经 credential helper，URL 不嵌 token）
git -c credential.helper="$CRED_HELPER" push <git_remote> daily_fix_<megatron_branch> --force
```

**检查清单**（推送前）：
- [ ] `pre-commit install` 已执行（hook 已安装到 `.git/hooks/pre-commit`）
- [ ] `pre-commit run --files <修复文件>` 全部 PASS（在 commit **之前**跑）
- [ ] `git log --oneline origin/<megatron_branch>..HEAD` 只有**一个 commit**
- [ ] commit author **和 committer** 均为配置中的 author（用 `git -c`，**不要**只用 `--author`——CLA 检查会失败）
- [ ] commit message 包含 `[branch: <megatron_branch>]` 标签
- [ ] `git remote -v` 无 `user:token@`（token 只存于 0600 的 `.push_config.json`，由 credential helper 提供）

## 创建 Issue + PR

push 成功后，由确定性脚本 `scripts/create_issue_pr.py --date <date>` 幂等创建（**LLM 不要手工 curl 创建**——历史教训：手工幂等查询 `search=<分支> <用例>` 含未编码空格返回空，每晚误判「无已存在 Issue」导致重复建 Issue）。→ API 细节见 `references/gitcode_api_guide.md`。

**空 PR 守卫（2026-08-24 新增）**：`create_issue_pr.py` 建 PR 前检查 `daily_fix_<分支>` 相对 `origin/<分支>` 是否有内容 diff（`fork_branch_has_diff`，tree 级 `git diff --quiet`）。无 diff（旧修复已合入 / 环境类修复无代码）→ 跳过建 Issue/PR，并对残留 stale open PR 评论提示人工关闭。前提：**⑦ 必须 `git checkout -B daily_fix_<分支> origin/<分支>` 从上游基线重建分支**（即使本次无新 fix），否则分支滞留 stale 提交。

**⚠️ 手动 / agent 驱动路径的前置条件**（cron needs_llm_diag 路径自动满足）：`create_issue_pr.py` 对每分支有两个门控——① `needs_llm_diag=true` **或** `output/<YYYY-MM>/<MM-DD>/<branch>/state/fixed_tests.json` 存在（create_issue_pr.py 2026-08-14 起改为 or 条件：fixed_tests.json 是"确认真问题"的设计信号，存在即处理）；② 有 `fixed_tests.json` / `diag_report.json` 才能取到失败用例 →「仅处理 PR」不建 Issue。手动修复路径默认缺 fixed_tests.json（2026-08-14 案例：修复走手动路径 → 只建了 PR #94、无 Issue），push 后需补写：

```bash
# 写 fixed_tests.json（只列确认修复的真问题测试名，排除 verify 误报）
python3 -c "
import json
json.dump(['test_<name>'], open('<state_dir>/fixed_tests.json', 'w'), ensure_ascii=False)
"
# 然后再跑幂等脚本（有 fixed_tests.json 即会处理该分支，无需改 needs_llm_diag）
python3 scripts/create_issue_pr.py --date <date>
```

脚本工作流（幂等保证）：
1. **按问题级去重**（2026-08-17 起）：问题 key = 归一化根因（diag_report `extracted_errors[].reason` → verify `failure_details` 按 test 解析 → `root_cause` 兜底），扫全量 open issues 标题归一化子串匹配 → 同一问题跨分支/用例/天只保留一个 Issue，**存在则追加「每日更新」评论**，不存在才新建（标题 `[daily-ci] <根因一句话>`，不再带分支/用例）。无根因的 unknown 问题退回 `[daily-ci] <branch> <test>` 前缀匹配复用既有 issue
2. 按 `<fork_user>:daily_fix_<分支>` 复用/创建 PR（fork_user 取自 `.push_config.json` 的 `repo.fork_path`；同一分支 force-push 自动更新 PR diff）
3. 评论 `compile` 触发 CI → 保存 PR 链接到 `output/<YYYY-MM>/<MM-DD>/<branch>/state/pr_info.json`
4. 每次新建 Issue 时写本地注册表 `output/issue_registry.jsonl`（append-only，记录 problem_key → issue number）

> ⚠️ gitcode API 的 `head=<user>:<branch>` 查询过滤实测不生效（返回空），脚本拉全量 open PR 后客户端按 `head.ref` 过滤。同样地，Issue 搜索返回后客户端按标题前缀二次过滤。

## TE-NPU 契约修复提交（手动 / agent 路径）

当 ⑥ DIAG 判定「**TE-NPU 偏离社区 TE 契约**」→ 修复落在 **TransformerEngineNPU 仓**（不是 MA `patches/`）。TE-NPU 是 Ascend 官方上游仓（`gitcode.com/Ascend/TransformerEngineNPU`），以 `{repo.author_name}` 身份提交到个人 fork，再对 Ascend 上游发 PR——流程与 MA 相同，但 **fork_path = `{repo.author_name}/TransformerEngineNPU`**（不在 `.push_config.json` 的 `repo.fork_path` 里，按 `{repo.author_name}/TransformerEngineNPU` 构造；token 复用 `repo.api_token`）。参考案例：FusedAdam step 序列化 → MR #136（分支 `fix/fused_adam_step_serialize_int`）。

**⚠️ 提交前先跑 TE-NPU 自有单测**：修复可能改变 TE-NPU 自己测试编码的旧契约（2026-08-17 案例：`test_adam_group_step_saved_and_loaded` 断言序列化 step 为 tensor，CI 失败；需随修复一起更新测试并本地验证）。跑 `python -m pytest tests/pytorch/test_fused_optimizer.py`（ma_agent_py312 + CANN）。

```bash
# ① 共享 clone 建分支 + 提交（身份复用 repo.author_*；git -c 同时设 author/committer，禁止 --author）
cd <workspace>/TransformerEngineNPU
git checkout -b fix/<descriptive>
git add <修改文件>
git -c user.name="$(python3 -c "import json;print(json.load(open('<skill_dir>/scripts/.push_config.json'))['repo']['author_name'])")" \
    -c user.email="$(python3 -c "import json;print(json.load(open('<skill_dir>/scripts/.push_config.json'))['repo']['author_email'])")" \
    commit -m "fix(<scope>): <desc>"

# ② 推送到个人 fork（URL 不嵌 token、不含个人用户名，全从 config 派生；认证经 credential helper）
TOKEN=$(python3 -c "import json;print(json.load(open('<skill_dir>/scripts/.push_config.json'))['repo']['api_token'])")  # 仅供下方 API 调用
FORK_USER=$(python3 -c "import json;print(json.load(open('<skill_dir>/scripts/.push_config.json'))['repo']['author_name'])")
CRED_HELPER="<skill_dir>/scripts/git-credential-helper.sh"
git remote remove fork 2>/dev/null || true
git remote add fork "https://gitcode.com/${FORK_USER}/TransformerEngineNPU.git"
git -c credential.helper="$CRED_HELPER" push -u fork fix/<descriptive>

# ③ 对 Ascend 上游发 PR（body 按 .gitcode/PULL_REQUEST_TEMPLATE.md 三 section 精简，合入即提交信息）
curl -s -X POST "https://gitcode.com/api/v5/repos/Ascend/TransformerEngineNPU/pulls" \
  -H "Content-Type: application/json" -H "PRIVATE-TOKEN: ${TOKEN}" \
  -d "$(python3 -c "import json;print(json.dumps({
    'head': '${FORK_USER}:fix/<descriptive>', 'base': 'main',
    'title': 'fix(<scope>): <desc>', 'body': '<三 section>',
    'fork_path': '${FORK_USER}/TransformerEngineNPU'}))")"
# 响应 .number → MR 编号, .web_url → 链接

# ④ 触发 CI + 关联 TE-NPU 仓 Issue（如建了）
curl -s -X POST "https://gitcode.com/api/v5/repos/Ascend/TransformerEngineNPU/pulls/<number>/comments" \
  -H "Content-Type: application/json" -H "PRIVATE-TOKEN: ${TOKEN}" -d '{"body": "compile"}'
```

要点：
- TE-NPU 共享 clone 始终 main；修复走**本地分支 + 推 fork**，提交后 `git checkout main` 还原工作树供 nightly 用
- **Ascend 合入前 nightly 仍靠 MA shim 兜底**（复发防护恢复）；合入后按 SKILL.md ⑦ 规则**移除 MA shim 防双写**
- gitcode **PATCH 关闭 Issue/PR 不可用**，superseded 的处理需手动在 web 关

## 编排入口

```bash
python3 scripts/nightly_orchestrator.py
```
