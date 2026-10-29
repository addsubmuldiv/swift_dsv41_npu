"""Compile and execute a small kernel to check the installed Ascend overlay."""
import importlib.metadata as metadata

import torch
import triton
import triton.language as tl


@triton.jit
def _add_one(x, y, size: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(x + offsets, offsets < size, other=0)
    tl.store(y + offsets, values + 1, offsets < size)


def verify_triton():
    x = torch.arange(256, device='npu', dtype=torch.float32)
    y = torch.empty_like(x)
    _add_one[(1,)](x, y, x.numel(), 256)
    torch.npu.synchronize()
    torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
    print('PASS: Triton-Ascend JIT on NPU;',
          'triton=' + metadata.distribution('triton').version,
          'triton-ascend=' + metadata.distribution('triton-ascend').version, flush=True)
