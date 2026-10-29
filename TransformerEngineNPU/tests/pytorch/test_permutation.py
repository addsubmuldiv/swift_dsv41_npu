# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

"""Unit tests for MoE permutation API"""

import pytest
import torch

from transformer_engine.pytorch.permutation import (
    moe_permute,
    moe_unpermute,
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
    moe_permute_with_probs,
    moe_permute_and_pad_with_probs,
)


def _seed() -> None:
    """Set random seed for reproducibility."""
    seed = 1234
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed(seed)


def _get_device() -> torch.device:
    """Return device based on availability (NPU if available, else CPU)."""
    return torch.device("npu" if torch.npu.is_available() else "cpu")


# ===================== Index Mode Tests =====================


class TestIndexMode:
    """Tests for permutation with index routing map."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        """Setup test fixtures."""
        _seed()
        self.num_tokens = 4
        self.hidden_size = 4
        self.topK = 2
        self.num_experts = 4

    def test_permute_unpermute_index_mode(self):
        """Test permute and unpermute with index mode, verify values and gradients."""
        # Create simple input: each row is [1, 1, 1, 1], [2, 2, 2, 2], etc.
        inp = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # token 0
                [2.0, 2.0, 2.0, 2.0],  # token 1
                [3.0, 3.0, 3.0, 3.0],  # token 2
                [4.0, 4.0, 4.0, 4.0],  # token 3
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        # Index: each token routes to 2 experts
        # token 0 -> expert 0, 1
        # token 1 -> expert 0, 1
        # token 2 -> expert 2, 3
        # token 3 -> expert 2, 3
        index = torch.tensor(
            [
                [0, 1],  # token 0 -> expert 0, 1
                [0, 1],  # token 1 -> expert 0, 1
                [2, 3],  # token 2 -> expert 2, 3
                [2, 3],  # token 3 -> expert 2, 3
            ],
            device=_get_device(),
            dtype=torch.int32,
        )

        # Simple probs: [0.3, 0.7] for each token
        probs = torch.tensor(
            [
                [0.3, 0.7],
                [0.3, 0.7],
                [0.3, 0.7],
                [0.3, 0.7],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        num_out_tokens = self.num_tokens * self.topK  # 8

        # ========== Permute ==========
        permuted, row_id_map = moe_permute(inp, index, num_out_tokens, map_type="index")

        # Verify permuted shape
        assert permuted.shape == (num_out_tokens, self.hidden_size), (
            f"Expected permuted shape ({num_out_tokens}, {self.hidden_size}), got {permuted.shape}"
        )
        assert row_id_map.shape == (num_out_tokens,), (
            f"Expected row_id_map shape ({num_out_tokens},), got {row_id_map.shape}"
        )

        # Verify permuted values: tokens should be grouped by expert
        # Expert 0: tokens 0, 1 -> rows [1,1,1,1], [2,2,2,2]
        # Expert 1: tokens 0, 1 -> rows [1,1,1,1], [2,2,2,2]
        # Expert 2: tokens 2, 3 -> rows [3,3,3,3], [4,4,4,4]
        # Expert 3: tokens 2, 3 -> rows [3,3,3,3], [4,4,4,4]
        # Expected permuted: [1,1,1,1], [2,2,2,2], [1,1,1,1], [2,2,2,2],
        # [3,3,3,3], [4,4,4,4], [3,3,3,3], [4,4,4,4]
        # (order within each expert group may vary, but expert groups are in order)
        expected_permuted = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # expert 0: token 0
                [2.0, 2.0, 2.0, 2.0],  # expert 0: token 1
                [1.0, 1.0, 1.0, 1.0],  # expert 1: token 0
                [2.0, 2.0, 2.0, 2.0],  # expert 1: token 1
                [3.0, 3.0, 3.0, 3.0],  # expert 2: token 2
                [4.0, 4.0, 4.0, 4.0],  # expert 2: token 3
                [3.0, 3.0, 3.0, 3.0],  # expert 3: token 2
                [4.0, 4.0, 4.0, 4.0],  # expert 3: token 3
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(permuted, expected_permuted, rtol=1e-5, atol=1e-5), (
            f"Permuted values incorrect.\nExpected:\n{expected_permuted}\nGot:\n{permuted}"
        )

        # ========== Unpermute ==========
        unpermuted = moe_unpermute(permuted, row_id_map, merging_probs=probs, map_type="index")

        # Verify unpermuted shape
        assert unpermuted.shape == (self.num_tokens, self.hidden_size), (
            f"Expected unpermuted shape ({self.num_tokens}, {self.hidden_size}), "
            f"got {unpermuted.shape}"
        )
        # Verify unpermuted values: each token should be weighted sum of its copies
        # token 0: 0.3 * [1,1,1,1] + 0.7 * [1,1,1,1] = [1,1,1,1]
        # token 1: 0.3 * [2,2,2,2] + 0.7 * [2,2,2,2] = [2,2,2,2]
        # token 2: 0.3 * [3,3,3,3] + 0.7 * [3,3,3,3] = [3,3,3,3]
        # token 3: 0.3 * [4,4,4,4] + 0.7 * [4,4,4,4] = [4,4,4,4]
        expected_unpermuted = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(unpermuted, expected_unpermuted, rtol=1e-5, atol=1e-5), (
            f"Unpermuted values incorrect.\nExpected:\n{expected_unpermuted}\nGot:\n{unpermuted}"
        )

        # ========== Gradient Test ==========
        inp_grad = inp.clone().detach().requires_grad_(True)
        probs_grad = probs.clone().detach().requires_grad_(True)

        # Forward pass
        permuted_g, row_id_map_g = moe_permute(inp_grad, index, num_out_tokens, map_type="index")
        # Simulate expert processing: multiply by 2
        permuted_processed = permuted_g * 2.0
        unpermuted_g = moe_unpermute(
            permuted_processed, row_id_map_g, merging_probs=probs_grad, map_type="index"
        )

        # Backward
        loss = unpermuted_g.sum()
        loss.backward()

        # Verify gradients exist and have correct shape
        assert inp_grad.grad is not None, "Input gradient should not be None"
        assert probs_grad.grad is not None, "Probs gradient should not be None"
        assert inp_grad.grad.shape == inp.shape, (
            f"Expected input grad shape {inp.shape}, got {inp_grad.grad.shape}"
        )
        assert probs_grad.grad.shape == probs.shape, (
            f"Expected probs grad shape {probs.shape}, got {probs_grad.grad.shape}"
        )

        # Verify gradient values
        # Let's trace the gradient flow:
        # loss = sum(unpermuted)
        # unpermuted[i] = sum_k(probs[i,k] * permuted_processed[k])
        # permuted_processed[k] = 2 * permuted[k]
        # permuted[k] = inp[token_k] (just reordering)
        #
        # So: d_loss/d_inp[i] = sum over all paths from inp[i] to loss
        # Each token appears in topK=2 experts
        # For token i, it contributes to unpermuted[i] through 2 experts
        # d_loss/d_inp[i] = 2 * (probs[i,0] + probs[i,1]) * hidden_size
        #                  = 2 * (0.3 + 0.7) * 4 = 8
        #
        # But wait, the actual gradient depends on how unpermute aggregates
        # Let me compute more carefully:
        # unpermuted[i] = probs[i,0] * permuted_processed[idx0]
        #                + probs[i,1] * permuted_processed[idx1]
        # where idx0, idx1 are the indices in permuted corresponding to token i's two experts
        # permuted_processed[idx] = 2 * inp[i]
        # So unpermuted[i] = 2 * inp[i] * (probs[i,0] + probs[i,1])
        # d_unpermuted[i] / d_inp[i] = 2 * (probs[i,0] + probs[i,1])
        # d_loss / d_inp[i] = sum over all j: d_loss/d_unpermuted[j] * d_unpermuted[j]/d_inp[i]
        # Since loss = sum(unpermuted), d_loss/d_unpermuted[j] = 1
        # So d_loss/d_inp[i] = sum over all j where inp[i] contributes to unpermuted[j]
        # Each inp[i] only contributes to unpermuted[i], so:
        # d_loss/d_inp[i] = 2 * (probs[i,0] + probs[i,1]) * hidden_size (for each element)
        #                  = 2 * 1.0 * 4 = 8 for each element
        # But the gradient is per-element, so each element gets gradient 2 * 1.0 = 2
        expected_inp_grad = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(inp_grad.grad, expected_inp_grad, rtol=1e-5, atol=1e-5), (
            f"Input gradient incorrect.\nExpected:\n{expected_inp_grad}\nGot:\n{inp_grad.grad}"
        )

        # For probs: gradient = sum of (permuted_token * 2) for each expert
        # d_loss/d_probs[i,k] = permuted_processed[idx] where idx is the index for token i, expert k
        # permuted_processed[idx] = 2 * inp[i]
        # So d_loss/d_probs[i,k] = 2 * inp[i] (summed over hidden_size)
        # probs[0, 0]: gradient = 2 * 1 * 4 = 8
        # probs[0, 1]: gradient = 2 * 1 * 4 = 8
        # probs[1, 0]: gradient = 2 * 2 * 4 = 16
        # probs[1, 1]: gradient = 2 * 2 * 4 = 16
        # probs[2, 0]: gradient = 2 * 3 * 4 = 24
        # probs[2, 1]: gradient = 2 * 3 * 4 = 24
        # probs[3, 0]: gradient = 2 * 4 * 4 = 32
        # probs[3, 1]: gradient = 2 * 4 * 4 = 32
        expected_probs_grad = torch.tensor(
            [
                [8.0, 8.0],
                [16.0, 16.0],
                [24.0, 24.0],
                [32.0, 32.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(probs_grad.grad, expected_probs_grad, rtol=1e-5, atol=1e-5), (
            f"Probs gradient incorrect.\nExpected:\n{expected_probs_grad}\nGot:\n{probs_grad.grad}"
        )

    def test_permute_with_zero_num_out_tokens(self):
        """Test permute with num_out_tokens=0, which should auto-infer as num_tokens * topK."""
        # Create simple input
        inp = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        # Index: each token routes to 2 experts
        index = torch.tensor(
            [
                [0, 1],
                [0, 1],
                [2, 3],
                [2, 3],
            ],
            device=_get_device(),
            dtype=torch.int32,
        )

        # Test with num_out_tokens=0 (should auto-infer)
        permuted, row_id_map = moe_permute(inp, index, num_out_tokens=0, map_type="index")

        # Verify that output shape is correctly inferred
        expected_num_out_tokens = self.num_tokens * self.topK  # 4 * 2 = 8
        assert permuted.shape == (expected_num_out_tokens, self.hidden_size), (
            f"Expected permuted shape ({expected_num_out_tokens}, {self.hidden_size}), "
            f"got {permuted.shape}"
        )
        assert row_id_map.shape == (expected_num_out_tokens,), (
            f"Expected row_id_map shape ({expected_num_out_tokens},), got {row_id_map.shape}"
        )

        # Verify that the result is the same as explicitly specifying num_out_tokens
        permuted_explicit, row_id_map_explicit = moe_permute(
            inp, index, num_out_tokens=expected_num_out_tokens, map_type="index"
        )

        assert torch.allclose(permuted, permuted_explicit, rtol=1e-5, atol=1e-5), (
            "Auto-inferred permute should match explicit permute"
        )
        assert torch.equal(row_id_map, row_id_map_explicit), (
            "Auto-inferred row_id_map should match explicit row_id_map"
        )


# ===================== Mask Mode Tests (Non-Pad) =====================


class TestMaskModeNonPad:
    """Tests for permutation with mask routing map (non-padded mode)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        """Setup test fixtures."""
        _seed()
        self.num_tokens = 4
        self.hidden_size = 4
        self.topK = 2
        self.num_experts = 4

    def test_permute_unpermute_mask_mode(self):
        """Test permute and unpermute with mask mode, verify values and gradients."""
        # Create simple input: each row is [1, 1, 1, 1], [2, 2, 2, 2], etc.
        inp = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # token 0
                [2.0, 2.0, 2.0, 2.0],  # token 1
                [3.0, 3.0, 3.0, 3.0],  # token 2
                [4.0, 4.0, 4.0, 4.0],  # token 3
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        # Routing map: binary mask indicating which experts each token goes to
        # token 0 -> expert 0, 1
        # token 1 -> expert 0, 1
        # token 2 -> expert 2, 3
        # token 3 -> expert 2, 3
        routing_map = torch.tensor(
            [
                [1, 1, 0, 0],  # token 0 -> expert 0, 1
                [1, 1, 0, 0],  # token 1 -> expert 0, 1
                [0, 0, 1, 1],  # token 2 -> expert 2, 3
                [0, 0, 1, 1],  # token 3 -> expert 2, 3
            ],
            device=_get_device(),
            dtype=torch.int8,
        )

        # Probs: [0.3, 0.7, 0, 0] for tokens 0,1 and [0, 0, 0.3, 0.7] for tokens 2,3
        probs = torch.tensor(
            [
                [0.3, 0.7, 0.0, 0.0],
                [0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.7],
                [0.0, 0.0, 0.3, 0.7],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        num_out_tokens = int(routing_map.sum().item())  # 8

        # ========== Permute ==========
        permuted, row_id_map = moe_permute(inp, routing_map, num_out_tokens, map_type="mask")

        # Verify permuted shape
        assert permuted.shape == (num_out_tokens, self.hidden_size), (
            f"Expected permuted shape ({num_out_tokens}, {self.hidden_size}), got {permuted.shape}"
        )
        assert row_id_map.shape == (num_out_tokens,), (
            f"Expected row_id_map shape ({num_out_tokens},), got {row_id_map.shape}"
        )

        # Verify permuted values: tokens should be grouped by expert
        # Expert 0: tokens 0, 1 -> rows [1,1,1,1], [2,2,2,2]
        # Expert 1: tokens 0, 1 -> rows [1,1,1,1], [2,2,2,2]
        # Expert 2: tokens 2, 3 -> rows [3,3,3,3], [4,4,4,4]
        # Expert 3: tokens 2, 3 -> rows [3,3,3,3], [4,4,4,4]
        expected_permuted = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # expert 0: token 0
                [2.0, 2.0, 2.0, 2.0],  # expert 0: token 1
                [1.0, 1.0, 1.0, 1.0],  # expert 1: token 0
                [2.0, 2.0, 2.0, 2.0],  # expert 1: token 1
                [3.0, 3.0, 3.0, 3.0],  # expert 2: token 2
                [4.0, 4.0, 4.0, 4.0],  # expert 2: token 3
                [3.0, 3.0, 3.0, 3.0],  # expert 3: token 2
                [4.0, 4.0, 4.0, 4.0],  # expert 3: token 3
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(permuted, expected_permuted, rtol=1e-5, atol=1e-5), (
            f"Permuted values incorrect.\nExpected:\n{expected_permuted}\nGot:\n{permuted}"
        )

        # ========== Unpermute ==========
        unpermuted = moe_unpermute(
            permuted,
            row_id_map,
            merging_probs=probs,
            restore_shape=(self.num_tokens, self.hidden_size),
            map_type="mask",
            routing_map=routing_map,
        )

        # Verify unpermuted shape
        assert unpermuted.shape == (self.num_tokens, self.hidden_size), (
            f"Expected unpermuted shape ({self.num_tokens}, {self.hidden_size}), "
            f"got {unpermuted.shape}"
        )

        # Verify unpermuted values
        expected_unpermuted = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(unpermuted, expected_unpermuted, rtol=1e-5, atol=1e-5), (
            f"Unpermuted values incorrect.\nExpected:\n{expected_unpermuted}\nGot:\n{unpermuted}"
        )

        # ========== Gradient Test ==========
        inp_grad = inp.clone().detach().requires_grad_(True)
        probs_grad = probs.clone().detach().requires_grad_(True)

        # Forward pass
        permuted_g, row_id_map_g = moe_permute(
            inp_grad, routing_map, num_out_tokens, map_type="mask"
        )
        permuted_processed = permuted_g * 2.0
        unpermuted_g = moe_unpermute(
            permuted_processed,
            row_id_map_g,
            merging_probs=probs_grad,
            restore_shape=(self.num_tokens, self.hidden_size),
            map_type="mask",
            routing_map=routing_map,
        )

        # Backward
        loss = unpermuted_g.sum()
        loss.backward()

        # Verify gradients exist and have correct shape
        assert inp_grad.grad is not None, "Input gradient should not be None"
        assert probs_grad.grad is not None, "Probs gradient should not be None"
        assert inp_grad.grad.shape == inp.shape
        assert probs_grad.grad.shape == probs.shape

        # Verify gradient values
        # Same logic as index mode: each element gets gradient 2 * (probs sum) = 2 * 1.0 = 2
        expected_inp_grad = torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(inp_grad.grad, expected_inp_grad, rtol=1e-5, atol=1e-5), (
            f"Input gradient incorrect.\nExpected:\n{expected_inp_grad}\nGot:\n{inp_grad.grad}"
        )

    def test_permute_with_probs_mask_mode(self):
        """Test permute_with_probs and unpermute with mask mode."""
        # Create simple input
        inp = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        routing_map = torch.tensor(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            device=_get_device(),
            dtype=torch.int8,
        )

        probs = torch.tensor(
            [
                [0.3, 0.7, 0.0, 0.0],
                [0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.7],
                [0.0, 0.0, 0.3, 0.7],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        num_out_tokens = int(routing_map.sum().item())

        # ========== Permute with probs ==========
        permuted, permuted_probs, row_id_map = moe_permute_with_probs(
            inp, probs, routing_map, num_out_tokens
        )

        # Verify shapes
        assert permuted.shape == (num_out_tokens, self.hidden_size)
        assert permuted_probs.shape == (num_out_tokens,)
        assert row_id_map.shape == (num_out_tokens,)

        # Verify permuted values: tokens should be grouped by expert
        expected_permuted = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # expert 0: token 0
                [2.0, 2.0, 2.0, 2.0],  # expert 0: token 1
                [1.0, 1.0, 1.0, 1.0],  # expert 1: token 0
                [2.0, 2.0, 2.0, 2.0],  # expert 1: token 1
                [3.0, 3.0, 3.0, 3.0],  # expert 2: token 2
                [4.0, 4.0, 4.0, 4.0],  # expert 2: token 3
                [3.0, 3.0, 3.0, 3.0],  # expert 3: token 2
                [4.0, 4.0, 4.0, 4.0],  # expert 3: token 3
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(permuted, expected_permuted, rtol=1e-5, atol=1e-5), (
            f"Permuted values incorrect.\nExpected:\n{expected_permuted}\nGot:\n{permuted}"
        )

        # Verify permuted_probs values: should match the routing
        # expert 0: probs[0,0]=0.3, probs[1,0]=0.3
        # expert 1: probs[0,1]=0.7, probs[1,1]=0.7
        # expert 2: probs[2,2]=0.3, probs[3,2]=0.3
        # expert 3: probs[2,3]=0.7, probs[3,3]=0.7
        expected_permuted_probs = torch.tensor(
            [0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.7, 0.7], device=_get_device(), dtype=torch.float32
        )
        assert torch.allclose(permuted_probs, expected_permuted_probs, rtol=1e-5, atol=1e-5), (
            f"Permuted probs incorrect.\nExpected:\n{expected_permuted_probs}\n"
            f"Got:\n{permuted_probs}"
        )
        # ========== Unpermute ==========
        unpermuted = moe_unpermute(
            permuted,
            row_id_map,
            merging_probs=probs,
            restore_shape=(self.num_tokens, self.hidden_size),
            map_type="mask",
            routing_map=routing_map,
        )

        # Verify unpermuted shape and values
        assert unpermuted.shape == (self.num_tokens, self.hidden_size)
        expected_unpermuted = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(unpermuted, expected_unpermuted, rtol=1e-5, atol=1e-5), (
            f"Unpermuted values incorrect.\nExpected:\n{expected_unpermuted}\nGot:\n{unpermuted}"
        )

        # ========== Gradient Test ==========
        inp_grad = inp.clone().detach().requires_grad_(True)
        probs_grad = probs.clone().detach().requires_grad_(True)

        permuted_g, permuted_probs_g, row_id_map_g = moe_permute_with_probs(
            inp_grad, probs_grad, routing_map, num_out_tokens
        )
        permuted_processed = permuted_g * 2.0
        unpermuted_g = moe_unpermute(
            permuted_processed,
            row_id_map_g,
            merging_probs=probs_grad,
            restore_shape=(self.num_tokens, self.hidden_size),
            map_type="mask",
            routing_map=routing_map,
        )

        loss = unpermuted_g.sum()
        loss.backward()

        assert inp_grad.grad is not None
        assert probs_grad.grad is not None
        assert inp_grad.grad.shape == inp.shape
        assert probs_grad.grad.shape == probs.shape


# ===================== Mask Mode Tests (Pad) =====================


class TestMaskModePad:
    """Tests for permutation with mask routing map (padded mode)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        """Setup test fixtures."""
        _seed()
        self.num_tokens = 4
        self.hidden_size = 4
        self.topK = 2
        self.num_experts = 4
        self.align_size = 2  # Alignment size for padding

    def test_permute_and_pad_with_probs(self):
        """Test permute_and_pad_with_probs and unpermute, verify values and gradients."""
        # Create simple input
        inp = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # token 0
                [2.0, 2.0, 2.0, 2.0],  # token 1
                [3.0, 3.0, 3.0, 3.0],  # token 2
                [4.0, 4.0, 4.0, 4.0],  # token 3
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        # Routing map with uneven distribution
        # token 0 -> expert 0
        # token 1 -> expert 0, 1
        # token 2 -> expert 2
        # token 3 -> expert 2, 3
        routing_map = torch.tensor(
            [
                [1, 0, 0, 0],  # token 0 -> expert 0
                [1, 1, 0, 0],  # token 1 -> expert 0, 1
                [0, 0, 1, 0],  # token 2 -> expert 2
                [0, 0, 1, 1],  # token 3 -> expert 2, 3
            ],
            device=_get_device(),
            dtype=torch.int8,
        )

        # Probs
        probs = torch.tensor(
            [
                [0.5, 0.0, 0.0, 0.0],
                [0.3, 0.7, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.3, 0.7],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        # Calculate tokens_per_expert
        tokens_per_expert = routing_map.sum(dim=0)  # [2, 1, 2, 1]

        # ========== Permute with padding ==========
        permuted, permuted_probs, row_id_map, pad_offsets, target_tokens_per_expert = (
            moe_permute_and_pad_with_probs(
                inp, probs, routing_map, tokens_per_expert, self.align_size
            )
        )

        # Verify shapes
        # tokens_per_expert = [2, 1, 2, 1]
        # target_tokens_per_expert = [2, 2, 2, 2] (aligned to 2)
        # total_padded_tokens = 8
        total_padded_tokens = target_tokens_per_expert.sum().item()
        assert permuted.shape == (total_padded_tokens, self.hidden_size), (
            f"Expected permuted shape ({total_padded_tokens}, {self.hidden_size}), "
            f"got {permuted.shape}"
        )
        assert permuted_probs.shape == (total_padded_tokens,)
        assert row_id_map.shape == (total_padded_tokens,)
        assert pad_offsets.shape == (self.num_experts,)

        # Verify target_tokens_per_expert is aligned
        expected_target = torch.tensor([2, 2, 2, 2], device=_get_device())
        assert torch.equal(target_tokens_per_expert, expected_target), (
            f"Expected target_tokens_per_expert {expected_target}, got {target_tokens_per_expert}"
        )

        # Verify pad_offsets
        # pad_lengths = [0, 1, 0, 1]
        # cum_pad = [0, 1, 1, 2]
        # pad_offsets = [0, 0, 1, 1]
        expected_pad_offsets = torch.tensor([0, 0, 1, 1], device=_get_device())
        assert torch.equal(pad_offsets, expected_pad_offsets), (
            f"Expected pad_offsets {expected_pad_offsets}, got {pad_offsets}"
        )

        # ========== Unpermute ==========
        unpermuted = moe_unpermute(
            permuted,
            row_id_map,
            merging_probs=probs,
            restore_shape=(self.num_tokens, self.hidden_size),
            map_type="mask",
            pad_offsets=pad_offsets,
            routing_map=routing_map,
        )

        # Verify unpermuted shape and values
        assert unpermuted.shape == (self.num_tokens, self.hidden_size)
        expected_unpermuted = torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [2.0, 2.0, 2.0, 2.0],
                [1.5, 1.5, 1.5, 1.5],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(unpermuted, expected_unpermuted, rtol=1e-5, atol=1e-5), (
            f"Unpermuted values incorrect.\nExpected:\n{expected_unpermuted}\nGot:\n{unpermuted}"
        )

        # ========== Gradient Test ==========
        inp_grad = inp.clone().detach().requires_grad_(True)
        probs_grad = probs.clone().detach().requires_grad_(True)

        permuted_g, permuted_probs_g, row_id_map_g, pad_offsets_g, target_tokens_g = (
            moe_permute_and_pad_with_probs(
                inp_grad, probs_grad, routing_map, tokens_per_expert, self.align_size
            )
        )
        permuted_processed = permuted_g * 2.0
        unpermuted_g = moe_unpermute(
            permuted_processed,
            row_id_map_g,
            merging_probs=probs_grad,
            restore_shape=(self.num_tokens, self.hidden_size),
            map_type="mask",
            pad_offsets=pad_offsets_g,
            routing_map=routing_map,
        )

        loss = unpermuted_g.sum()
        loss.backward()

        assert inp_grad.grad is not None
        assert probs_grad.grad is not None
        assert inp_grad.grad.shape == inp.shape
        assert probs_grad.grad.shape == probs.shape


# ===================== Sort Chunks Tests =====================


class TestSortChunks:
    """Tests for sort chunks by index."""

    @pytest.fixture(autouse=True)
    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        """Setup test fixtures."""
        _seed()
        self.num_tokens = 8
        self.hidden_size = 4

    @pytest.mark.skip(reason="Hanged to be fixed")
    def test_sort_chunks_by_index(self):
        """Test sort chunks by index, verify values and gradients."""
        # Create input: 8 tokens, each row is [1,1,1,1], [2,2,2,2], etc.
        inp = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],  # chunk 0
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],  # chunk 1
                [4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0],  # chunk 2
                [6.0, 6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0, 7.0],  # chunk 3
                [8.0, 8.0, 8.0, 8.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        # Split sizes: 4 chunks of size 2 each
        split_sizes = torch.tensor([2, 2, 2, 2], device=_get_device(), dtype=torch.int32)

        # Sorted index: reorder chunks as [2, 0, 3, 1]
        sorted_index = torch.tensor([2, 0, 3, 1], device=_get_device(), dtype=torch.int32)

        # ========== Sort chunks ==========
        output = moe_sort_chunks_by_index(inp, split_sizes, sorted_index)

        # Verify output shape
        assert output.shape == inp.shape, f"Expected output shape {inp.shape}, got {output.shape}"

        # Verify output values: chunks should be reordered
        # Original: chunk 0 = [1,2], chunk 1 = [3,4], chunk 2 = [5,6], chunk 3 = [7,8]
        # After sort: chunk 2, chunk 0, chunk 3, chunk 1
        # Expected: [5,6, 1,2, 7,8, 3,4]
        expected_output = torch.tensor(
            [
                [5.0, 5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0, 6.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [7.0, 7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0, 8.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-5), (
            f"Output values incorrect.\nExpected:\n{expected_output}\nGot:\n{output}"
        )

        # ========== Gradient Test ==========
        inp_grad = inp.clone().detach().requires_grad_(True)

        output_g = moe_sort_chunks_by_index(inp_grad, split_sizes, sorted_index)
        output_processed = output_g * 2.0

        loss = output_processed.sum()
        loss.backward()

        assert inp_grad.grad is not None
        assert inp_grad.grad.shape == inp.shape

        # Verify gradient values: each element should have gradient 2.0
        expected_grad = torch.full_like(inp, 2.0)
        assert torch.allclose(inp_grad.grad, expected_grad, rtol=1e-5, atol=1e-5), (
            f"Gradient values incorrect.\nExpected:\n{expected_grad}\nGot:\n{inp_grad.grad}"
        )

    @pytest.mark.skip(reason="Hanged to be fixed")
    def test_sort_chunks_by_index_with_probs(self):
        """Test sort chunks by index with probs, verify values and gradients."""
        # Create input
        inp = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0, 8.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )

        # Simple probs: [0.3, 0.7, 0.3, 0.7, ...]
        probs = torch.tensor(
            [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7], device=_get_device(), dtype=torch.float32
        )

        split_sizes = torch.tensor([2, 2, 2, 2], device=_get_device(), dtype=torch.int32)
        sorted_index = torch.tensor([2, 0, 3, 1], device=_get_device(), dtype=torch.int32)

        # ========== Sort chunks with probs ==========
        output, permuted_probs = moe_sort_chunks_by_index_with_probs(
            inp, probs, split_sizes, sorted_index
        )

        # Verify shapes
        assert output.shape == inp.shape
        assert permuted_probs.shape == (self.num_tokens,)

        # Verify output values
        expected_output = torch.tensor(
            [
                [5.0, 5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0, 6.0],
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [7.0, 7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0, 8.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            device=_get_device(),
            dtype=torch.float32,
        )
        assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-5), (
            f"Output values incorrect.\nExpected:\n{expected_output}\nGot:\n{output}"
        )

        # Verify permuted_probs: should be reordered same as tokens
        expected_probs = torch.tensor(
            [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7], device=_get_device(), dtype=torch.float32
        )
        assert torch.allclose(permuted_probs, expected_probs, rtol=1e-5, atol=1e-5), (
            f"Permuted probs incorrect.\nExpected:\n{expected_probs}\nGot:\n{permuted_probs}"
        )

        # ========== Gradient Test ==========
        inp_grad = inp.clone().detach().requires_grad_(True)
        probs_grad = probs.clone().detach().requires_grad_(True)

        output_g, permuted_probs_g = moe_sort_chunks_by_index_with_probs(
            inp_grad, probs_grad, split_sizes, sorted_index
        )
        output_processed = output_g * 2.0

        loss = output_processed.sum()
        loss.backward()

        assert inp_grad.grad is not None
        assert probs_grad.grad is not None
        assert inp_grad.grad.shape == inp.shape
        assert probs_grad.grad.shape == probs.shape


# ===================== Edge Cases Tests =====================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        _seed()

    def test_empty_input_index_mode(self):
        """Test with empty input in index mode."""
        num_tokens = 0
        hidden_size = 4
        topK = 2

        inp = torch.empty((num_tokens, hidden_size), device=_get_device(), dtype=torch.float32)
        index = torch.empty((num_tokens, topK), device=_get_device(), dtype=torch.int32)
        probs = torch.empty((num_tokens, topK), device=_get_device(), dtype=torch.float32)

        # Permute
        permuted, row_id_map = moe_permute(inp, index, 0, map_type="index")

        # Verify shapes
        assert permuted.shape == (0, hidden_size)
        assert row_id_map.shape == (0,)

        # Unpermute
        unpermuted = moe_unpermute(permuted, row_id_map, merging_probs=probs, map_type="index")
        assert unpermuted.shape == (0, hidden_size)

    def test_empty_input_mask_mode(self):
        """Test with empty input in mask mode."""
        num_tokens = 0
        hidden_size = 4
        num_experts = 4

        inp = torch.empty((num_tokens, hidden_size), device=_get_device(), dtype=torch.float32)
        routing_map = torch.empty((num_tokens, num_experts), device=_get_device(), dtype=torch.int8)
        probs = torch.empty((num_tokens, num_experts), device=_get_device(), dtype=torch.float32)

        # Permute
        permuted, row_id_map = moe_permute(inp, routing_map, 0, map_type="mask")

        # Verify shapes
        assert permuted.shape == (0, hidden_size)
        assert row_id_map.shape == (0,)

        # Unpermute
        unpermuted = moe_unpermute(
            permuted,
            row_id_map,
            merging_probs=probs,
            restore_shape=(num_tokens, hidden_size),
            map_type="mask",
            routing_map=routing_map,
        )
        assert unpermuted.shape == (0, hidden_size)

    def test_single_token_index_mode(self):
        """Test with single token in index mode."""
        hidden_size = 4
        topK = 2

        inp = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=_get_device(), dtype=torch.float32)
        index = torch.tensor([[0, 1]], device=_get_device(), dtype=torch.int32)
        probs = torch.tensor([[0.3, 0.7]], device=_get_device(), dtype=torch.float32)

        # Permute
        permuted, row_id_map = moe_permute(inp, index, topK, map_type="index")

        # Verify shapes
        assert permuted.shape == (topK, hidden_size)
        assert row_id_map.shape == (topK,)

        # Verify permuted values: both rows should be [1,1,1,1]
        assert torch.allclose(permuted[0], inp[0], rtol=1e-5, atol=1e-5)
        assert torch.allclose(permuted[1], inp[0], rtol=1e-5, atol=1e-5)

        # Unpermute
        unpermuted = moe_unpermute(permuted, row_id_map, merging_probs=probs, map_type="index")

        # Verify shape and value
        assert unpermuted.shape == (1, hidden_size)
        expected = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=_get_device(), dtype=torch.float32)
        assert torch.allclose(unpermuted, expected, rtol=1e-5, atol=1e-5)

    def test_single_token_mask_mode(self):
        """Test with single token in mask mode."""
        hidden_size = 4

        inp = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=_get_device(), dtype=torch.float32)
        routing_map = torch.tensor([[1, 1, 0, 0]], device=_get_device(), dtype=torch.int8)
        probs = torch.tensor([[0.3, 0.7, 0.0, 0.0]], device=_get_device(), dtype=torch.float32)

        num_out_tokens = 2

        # Permute
        permuted, row_id_map = moe_permute(inp, routing_map, num_out_tokens, map_type="mask")

        # Verify shapes
        assert permuted.shape == (num_out_tokens, hidden_size)
        assert row_id_map.shape == (num_out_tokens,)

        # Verify permuted values
        assert torch.allclose(permuted[0], inp[0], rtol=1e-5, atol=1e-5)
        assert torch.allclose(permuted[1], inp[0], rtol=1e-5, atol=1e-5)

        # Unpermute
        unpermuted = moe_unpermute(
            permuted,
            row_id_map,
            merging_probs=probs,
            restore_shape=(1, hidden_size),
            map_type="mask",
            routing_map=routing_map,
        )

        # Verify shape and value
        assert unpermuted.shape == (1, hidden_size)
        expected = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=_get_device(), dtype=torch.float32)
        assert torch.allclose(unpermuted, expected, rtol=1e-5, atol=1e-5)

    def test_invalid_map_type(self):
        """Test with invalid map_type."""
        inp = torch.randn(4, 4, device=_get_device(), dtype=torch.float32)
        routing_map = torch.zeros(4, 4, device=_get_device(), dtype=torch.int8)

        with pytest.raises(ValueError, match="map_type should be one of"):
            moe_permute(inp, routing_map, 8, map_type="invalid")

    def test_different_dtypes(self):
        """Test with different data types."""
        num_tokens = 4
        topK = 2

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            inp = torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0, 4.0],
                ],
                device=_get_device(),
                dtype=dtype,
            )
            index = torch.tensor(
                [
                    [0, 1],
                    [0, 1],
                    [2, 3],
                    [2, 3],
                ],
                device=_get_device(),
                dtype=torch.int32,
            )
            probs = torch.tensor(
                [
                    [0.3, 0.7],
                    [0.3, 0.7],
                    [0.3, 0.7],
                    [0.3, 0.7],
                ],
                device=_get_device(),
                dtype=torch.float32,
            )

            # Permute
            permuted, row_id_map = moe_permute(inp, index, num_tokens * topK, map_type="index")

            # Verify dtype is preserved
            assert permuted.dtype == dtype, f"Expected dtype {dtype}, got {permuted.dtype}"

            # Unpermute
            unpermuted = moe_unpermute(permuted, row_id_map, merging_probs=probs, map_type="index")

            # Verify dtype
            assert unpermuted.dtype == dtype, f"Expected dtype {dtype}, got {unpermuted.dtype}"
