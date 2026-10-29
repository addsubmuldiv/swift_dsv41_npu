# Local validation

## Initial validation (Triton-Ascend 3.6.0)

Validated on 2026-10-29 with the source snapshots and py311 patches in this repository.
A separate Python 3.11 environment was used for this checkout.

- Python 3.11.18, torch 2.10.0+cpu, torch_npu 2.10.0.post7, CANN 9.2.0.
- Ran `bash setup_deepseek_v41_npu.sh`: exit 0; `pip check` passed.
- All five source modules loaded from this checkout; NPU matmul and DeepSeek V4.1 initialization passed.
- Ran `examples/smoke_train.sh` against local DeepSeek-V4.1-Flash-1017 weights.
- Four original layers, original width/384 experts/first Engram, 6 NPUs, EP6/TP1/PP1/CP1, BF16 LoRA rank8/alpha16.
- Completed five forward/backward/optimizer steps and saved Megatron/safetensors checkpoints; exit 0.
- Each of the 6 ranks checked that all five source components came from this checkout.
- A normal clone to a different directory contained all 5900 source entries, without submodules.
- Ran the installer from that cloned directory: exit 0; all five module paths resolved inside the clone.

| Step | Reported training loss | Gradient norm |
| --- | --- | --- |
| 1 | 13.2276659 | 1.74964523 |
| 2 | 13.17769241 | 1.84241819 |
| 3 | 13.31243324 | 2.39051318 |
| 4 | 13.63906479 | 3.90207195 |
| 5 | 12.5469141 | 4.10004902 |

This verifies installation and a reduced training path. It is not full-model, multi-node,
or 200-step precision validation. No model weights or machine-specific logs are published.
See `validation.json` for the structured summary.

## Triton-Ascend 3.2.2 follow-up

The installer now defaults to Triton-Ascend 3.2.2, not the 3.6.0 baseline above.
Official cp311 wheel metadata requires community Triton 3.2.0 on x86_64,
Triton 3.5.0 on aarch64, and pytest 8.3.2 on both.

- x86_64 installation, pip check, source paths, NPU initialization: passed.
- NPU Triton JIT compilation/execution and TENPU Linear/LayerNormLinear/GroupedLinear forward/backward: passed.
- Six-NPU, four-layer, five-step LoRA training and checkpoint/safetensors saving: passed.
- aarch64 cp311 Ascend wheel: downloaded and its actual metadata checked.
- aarch64 Triton/Ascend dependency resolution: passed with the platform-specific constraints.
- Two architecture-constraint regression tests: passed.
- aarch64 hardware training was not run on this x86_64 host. No new 200-step precision comparison was run.

The installer explicitly obtains the Ascend wheel from its dedicated index even
when the container's `PIP_EXTRA_INDEX_URL` points elsewhere. It restores the Ascend
shared-file overlay after community Triton installation, and verifies an NPU JIT kernel.
Historical 3.6.0 precision results are not relabeled as 3.2.2 results.
