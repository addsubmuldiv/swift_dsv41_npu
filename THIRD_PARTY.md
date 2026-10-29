# Third-party source provenance

This repository vendors fixed snapshots of five projects, including their original LICENSE,
copyright notices, documentation and tests. It does not replace those projects' licenses
with a single repository-wide license. Consult each source directory's LICENSE.

| Directory | Source | Base commit | Local changes |
| --- | --- | --- | --- |
| ms-swift | https://github.com/addsubmuldiv/ms-swift | a013ae639060ba3439a60dd7cb07ff9561d06be6 | None |
| mcore-bridge | https://github.com/addsubmuldiv/mcore-bridge | e7cb294a24cf98ac16be85774646cc8f46f9f6c2 | patches/bridge-optional-triton.patch, already applied |
| Megatron-LM | https://github.com/NVIDIA/Megatron-LM | 5c6efbc278b708c046e5b558964cb021f6fd750b | patches/mcore-python311.patch and patches/mcore-optional-triton.patch, already applied |
| MegatronAdaptor | https://gitcode.com/Ascend/MegatronAdaptor | c7e52b2c8482105e6ec7fcd8cdef364e7c7b22c5 | None |
| TransformerEngineNPU | https://gitcode.com/Ascend/TransformerEngineNPU | 93eca54f264510f31dd10b68cd8ae18ad9a96bbe | patches/tenpu-python311.patch and patches/tenpu-optional-triton.patch, already applied |

The integration scripts and synthetic smoke dataset are provided alongside these snapshots.
Model weights, tokenizer artifacts, Conda environments, caches and local machine logs are not distributed.
