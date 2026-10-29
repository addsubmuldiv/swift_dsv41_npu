# DeepSeek V4.1 Flash NPU 环境

五个仓库的固定版本源码已直接包含在本仓库内，Python 3.11 补丁也已应用。
普通 `git clone` 即可取得全部源码，无需 submodule、逐仓库切分支或手动打补丁。

## 安装

在每台机器的容器中分别运行，使用该容器当前的 Python 环境：

```bash
git clone https://github.com/addsubmuldiv/swift_dsv41_npu.git
cd swift_dsv41_npu
bash setup_deepseek_v41_npu.sh
source env.sh
```

如果四台机器共用同一份源码，只需克隆一次，在各自容器中执行安装脚本。
脚本通过文件锁串行处理共享源码的 editable 安装。
每个新训练终端都需要 `source env.sh`；路径根据当前目录计算，可移动到其他路径。

基础镜像需要已安装 Python 3.11、Torch/TorchNPU 2.10，并能访问系统 CANN 和 NPU 驱动。
本地验证使用 Python 3.11.18、torch 2.10.0+cpu、torch_npu 2.10.0.post7、CANN 9.2.0。
Python 3.12 也允许安装；本仓库的新安装与训练验证使用 3.11。
安装脚本不创建 Conda、不下载模型、不启动训练，也不会替换已安装的 Torch/TorchNPU。
基础版本不符合要求时会直接报错。

脚本会安装锁定版本的 Python 依赖，再 editable 安装五个本地目录，最后执行依赖检查、
源码加载路径检查、NPU 张量运算及 DeepSeek V4.1 注册/初始化检查。
默认不安装 Triton / Triton-Ascend，也不预取 Ascend wheel、不强制使用华为源。
依赖下载沿用容器的 pip 配置；可通过 `PIP_INDEX_URL` / `PIP_EXTRA_INDEX_URL` 指定可达的镜像。
当前示例使用 BF16、非融合 attention，不需要 Triton；MXFP8 和部分可选融合算子仍可能需要它。
仅在明确需要这些算子时，另行准备可达的 wheel 源，使用 `constraints-triton.txt` 中的约束
安装 `TransformerEngineNPU[triton]`（Ascend 3.2.2）；默认安装脚本不执行此步骤。
如果容器里有内部标签不兼容的可选旧 decord wheel，脚本会移除它；本次只验证语言训练。

## 单机减层训练验证

已有 DeepSeek-V4.1-Flash-1017 权重时：

```bash
source env.sh
MODEL_DIR=/path/to/DeepSeek-V4.1-Flash-1017 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 \
bash examples/smoke_train.sh
```

此示例保留原始第 0–3 层、原宽度、384 专家及首个 Engram，使用 6 卡 EP6、BF16 LoRA，
训练 5 步并保存 checkpoint/safetensors。它生成缩减配置与权重索引，用软链接复用原始权重分片，
不复制完整权重。Engram tokenizer map 使用权重目录内官方的两个辅助函数生成。
只对可信的本地权重目录运行。示例默认需要每张约 64 GB 的 NPU，运行前确保六张卡空闲。

生成的模型视图、训练结果放在 `.local/`，不纳入 Git；可通过 `SMOKE_MODEL`、`SMOKE_OUTPUT`
覆盖位置。这个示例仅验证安装与训练链路，不是完整模型训练或通用精度认证。
多机正式训练仍需按任务配置节点数、rank、master 地址和训练数据。

## 四机完整模型 LoRA 训练

4 台 × 每台 16 张 64GB NPU 的启动脚本为
[`examples/train_full_lora_4nodes.sh`](examples/train_full_lora_4nodes.sh)。
保留全部 40 层、384 专家和两个 Engram，默认 TP1/PP4/EP16。
使用仓库自带测试数据，在四台机器分别启动 rank 0–3；
具体命令和参数见 [四机启动说明](examples/README_full_lora_4nodes.md)。
此脚本已验证 Engram map 生成与参数解析，尚未实测四机完整模型训练。

## 完整环境与精度说明

参见 [DeepSeek V4.1 Flash：NPU 软件环境、补丁与 GPU 精度验证](docs/DeepSeek_V41_Flash_NPU_环境与精度验证.md)，包含版本矩阵、安装流程、实际报错及补丁、200 步对比方法与结果边界。

## 验证记录

本地一键安装、普通克隆后的再次安装、6 卡 5 步训练及保存均已通过，见 [VALIDATION.md](VALIDATION.md)。
不包含多机或完整模型训练验证。

## 源码与补丁

- `ms-swift/`：addsubmuldiv fork，固定 `a013ae639060ba3439a60dd7cb07ff9561d06be6`。
- `mcore-bridge/`：addsubmuldiv fork，固定 `e7cb294a24cf98ac16be85774646cc8f46f9f6c2`，含 NPU/Engram 适配和设备反量化。
- `Megatron-LM/`：DeepSeek 专用版本 `5c6efbc278b708c046e5b558964cb021f6fd750b`，不是普通 Core 0.19 的任意分支。
- `MegatronAdaptor/`：`c7e52b2c8482105e6ec7fcd8cdef364e7c7b22c5`。
- `TransformerEngineNPU/`：`93eca54f264510f31dd10b68cd8ae18ad9a96bbe`。

已应用的 py311 补丁：TENPU 安装最低版本调整；MCore 两处 `typing.override`
改用 `typing_extensions.override`。补丁副本放在 `patches/`，安装时不再应用。
依赖使用 `av==18.1.0`、`networkx==3.6.1`，避免旧环境中要求 Python 3.12 的版本。

另已应用可选 Triton 补丁：TENPU 将 Triton 移到 extra 并延迟加载 MXFP8 内核；
MCore 延迟加载 paged-stash 内核；bridge 将 QSA 内核导入放进已有的可用性检查。
这样未启用这些功能时可以不安装 Triton。补丁同样位于 `patches/`。

完整来源见 `sources.lock.json`；预置源码文件摘要见 `source-files.sha256.json`
（普通文件散列内容，软链接散列链接目标文本）。各项目保留原始 LICENSE，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
