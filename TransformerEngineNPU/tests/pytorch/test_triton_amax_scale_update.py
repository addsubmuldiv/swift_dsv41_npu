# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numerical parity test for the fused Triton amax and scale update."""

import pytest
import torch


pytest.importorskip("torch_npu")
pytest.importorskip("triton")

from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.pytorch.quantization import is_fp8_available
from transformer_engine.pytorch.quantization.manager import _amax_and_scale_update
from transformer_engine.pytorch.triton.amax_scale_update import fused_amax_and_scale_update

pytestmark = pytest.mark.skipif(
    not is_fp8_available(),
    reason="quantized tests require Atlas A5",
)


def _make_pairs(history_len, quantizers, fp8_max):
    histories = []
    scales = []
    offset = 0
    fp32_max = torch.finfo(torch.float32).max
    tiny_amax = fp8_max / fp32_max / 2.0

    for num_quantizers in quantizers:
        num_values = history_len * num_quantizers
        history = (
            torch.arange(offset, offset + num_values, dtype=torch.float32)
            .reshape(history_len, num_quantizers)
            .remainder(31)
            .add_(1.0)
        )
        scale = torch.arange(1, num_quantizers + 1, dtype=torch.float32).add_(2.0)

        history[:, 0].zero_()
        if num_quantizers > 1:
            history[:, 1].fill_(tiny_amax)
        if num_quantizers > 3:
            history[:, 3].fill_(torch.inf)

        histories.append(history.npu())
        scales.append(scale.npu())
        offset += num_values

    return histories, scales


@pytest.mark.parametrize("algo", ["max", "mean", "most_recent"])
@pytest.mark.parametrize(
    ("history_len", "quantizers"),
    [
        (1, [5]),
        (3, [5]),
        (127, [2, 5]),
        (1024, [4, 1, 3]),
    ],
)
def test_fused_update_matches_small_operator(algo, history_len, quantizers):
    fp8_max = 448.0
    recipe = DelayedScaling(
        amax_history_len=history_len,
        amax_compute_algo=algo,
        margin=1,
    )
    histories, scales = _make_pairs(history_len, quantizers, fp8_max)
    reference_histories = [history.clone() for history in histories]
    reference_scales = [scale.clone() for scale in scales]

    fused_amax_and_scale_update(histories, scales, fp8_max, recipe)
    for history, scale in zip(reference_histories, reference_scales):
        _amax_and_scale_update(history, scale, fp8_max, recipe)

    for history, scale, reference_history, reference_scale in zip(
        histories,
        scales,
        reference_histories,
        reference_scales,
    ):
        torch.testing.assert_close(
            history,
            reference_history,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            scale,
            reference_scale,
            rtol=2e-5,
            atol=1e-6,
        )
