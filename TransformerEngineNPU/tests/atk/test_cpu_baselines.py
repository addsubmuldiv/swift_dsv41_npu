"""CPU-only regression tests for ATK references; no ATK installation required."""

import importlib.util
from pathlib import Path
import unittest

import torch


_SPEC = importlib.util.spec_from_file_location(
    "atk_cpu_baselines", Path(__file__).parent / "module" / "_cpu_baselines.py"
)
_BASELINES = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASELINES)


class UnpermuteMaskDtypeTest(unittest.TestCase):
    def test_no_padding_round_trip_and_gradients(self):
        # Balanced loads make fixed-capacity permute and unpadded unpermute
        # equivalent, isolating mask dtype from backend layout differences.
        routes = (
            [[1, 0], [0, 1], [1, 0], [0, 1]],
            [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        )
        for rows in routes:
            for mask_dtype in (torch.bool, torch.int8, torch.int32):
                with self.subTest(topk=sum(rows[0]), mask_dtype=mask_dtype):
                    mask = torch.tensor(rows, dtype=mask_dtype)
                    tokens, experts = mask.shape
                    inp = torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)
                    inp.requires_grad_()
                    probs = torch.arange(1, tokens * experts + 1, dtype=torch.float32)
                    probs = (probs.reshape(tokens, experts) / 10).requires_grad_()
                    dy = torch.arange(1, tokens * 2 + 1, dtype=torch.float32).reshape(tokens, 2) / 7
                    counts = mask.to(torch.int32).sum(0).to(torch.int32)

                    permuted, _, row_map, offsets, _ = _BASELINES.mp_pad_probs_cpu(
                        inp, probs, mask, counts, align_size=1
                    )
                    self.assertIsNone(offsets)
                    actual = _BASELINES.moe_unpermute_pad_probs_cpu(
                        permuted,
                        row_map,
                        merging_probs=probs,
                        restore_shape=inp.shape,
                        pad_offsets=offsets,
                        routing_map=mask,
                    )
                    expected = inp * (probs * mask.bool()).sum(1, keepdim=True)
                    torch.testing.assert_close(actual, expected)
                    actual_grads = torch.autograd.grad((actual * dy).sum(), (inp, probs))
                    expected_grads = torch.autograd.grad((expected * dy).sum(), (inp, probs))
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
