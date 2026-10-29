# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Test router functions for NPU"""

import pytest
import torch

from transformer_engine.pytorch.router import (
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
)


def test_fused_topk_with_score_function_softmax():
    """Test fused_topk_with_score_function with softmax score function."""
    num_tokens = 4
    num_experts = 4
    topk = 2

    # Create predictable logits for easy verification
    # Each row has increasing values, so topk should select the last 2 experts
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Test with use_pre_softmax=False (topk first, then softmax)
    probs, routing_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
    )

    # Check shapes
    assert probs.shape == (num_tokens, num_experts)
    assert routing_map.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Check that probs are non-zero only for selected experts
    assert (probs[~routing_map] == 0).all()

    # Verify specific values for first token
    # First token: logits = [1.0, 2.0, 3.0, 4.0], topk=2 should select indices [3, 2]
    # Selected logits: [4.0, 3.0]
    # Softmax: exp(4.0)/(exp(4.0)+exp(3.0)) ≈ 0.731, exp(3.0)/(exp(4.0)+exp(3.0)) ≈ 0.269
    expected_probs_0 = torch.softmax(torch.tensor([4.0, 3.0]), dim=0)
    assert torch.allclose(probs[0, 3], expected_probs_0[0], atol=1e-5)
    assert torch.allclose(probs[0, 2], expected_probs_0[1], atol=1e-5)
    assert probs[0, 0] == 0.0
    assert probs[0, 1] == 0.0

    # Verify routing_map for first token
    assert routing_map[0, 2].item()  # expert 2 selected
    assert routing_map[0, 3].item()  # expert 3 selected
    assert not routing_map[0, 0].item()
    assert not routing_map[0, 1].item()


def test_fused_topk_with_score_function_sigmoid():
    """Test fused_topk_with_score_function with sigmoid score function."""
    num_tokens = 4
    num_experts = 4
    topk = 2

    # Create predictable logits
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Test with sigmoid
    probs, routing_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="sigmoid",
        expert_bias=None,
    )

    # Check shapes
    assert probs.shape == (num_tokens, num_experts)
    assert routing_map.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Verify specific values for first token
    # First token: logits = [1.0, 2.0, 3.0, 4.0]
    # Sigmoid scores: [sigmoid(1), sigmoid(2), sigmoid(3), sigmoid(4)]
    sigmoid_scores = torch.sigmoid(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # topk=2 should select indices [3, 2] (largest sigmoid values)
    selected_scores = sigmoid_scores[torch.tensor([3, 2])]
    # Normalize: selected_scores / sum(selected_scores)
    expected_probs_0 = selected_scores / selected_scores.sum()

    assert torch.allclose(probs[0, 3], expected_probs_0[0], atol=1e-5)
    assert torch.allclose(probs[0, 2], expected_probs_0[1], atol=1e-5)
    assert probs[0, 0] == 0.0
    assert probs[0, 1] == 0.0


def test_fused_topk_with_score_function_sqrtsoftplus():
    """Test fused_topk_with_score_function with sqrtsoftplus score function."""
    num_tokens = 4
    num_experts = 4
    topk = 2

    # Create predictable logits for easy verification
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Test with sqrtsoftplus
    probs, routing_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="sqrtsoftplus",
        expert_bias=None,
    )

    # Check shapes
    assert probs.shape == (num_tokens, num_experts)
    assert routing_map.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Verify specific values for first token
    # First token: logits = [1.0, 2.0, 3.0, 4.0]
    # sqrtsoftplus scores: [sqrt(softplus(1)), sqrt(softplus(2)), sqrt(softplus(3)), sqrt(softplus(4))]
    sqrtsoftplus_scores = torch.nn.functional.softplus(torch.tensor([1.0, 2.0, 3.0, 4.0])).sqrt()
    # topk=2 should select indices [3, 2] (largest sqrtsoftplus values)
    selected_scores = sqrtsoftplus_scores[torch.tensor([3, 2])]
    # Normalize: selected_scores / sum(selected_scores)
    expected_probs_0 = selected_scores / selected_scores.sum()

    assert torch.allclose(probs[0, 3], expected_probs_0[0], atol=1e-5)
    assert torch.allclose(probs[0, 2], expected_probs_0[1], atol=1e-5)
    assert probs[0, 0] == 0.0
    assert probs[0, 1] == 0.0

    # Verify routing_map for first token
    assert routing_map[0, 2].item()  # expert 2 selected
    assert routing_map[0, 3].item()  # expert 3 selected
    assert not routing_map[0, 0].item()
    assert not routing_map[0, 1].item()


def test_fused_topk_with_score_function_with_scaling():
    """Test fused_topk_with_score_function with scaling factor."""
    num_tokens = 4
    num_experts = 4
    topk = 2
    scaling_factor = 0.5

    # Create predictable logits for easy verification
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Test with scaling factor
    probs, routing_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=scaling_factor,
        score_function="softmax",
        expert_bias=None,
    )

    # Check shapes
    assert probs.shape == (num_tokens, num_experts)
    assert routing_map.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Verify specific values for first token with scaling
    # First token: logits = [1.0, 2.0, 3.0, 4.0], topk=2 should select indices [3, 2]
    # Selected logits: [4.0, 3.0]
    # Softmax: exp(4.0)/(exp(4.0)+exp(3.0)) ≈ 0.731, exp(3.0)/(exp(4.0)+exp(3.0)) ≈ 0.269
    # With scaling factor 0.5: probs should be scaled by 0.5
    expected_probs_0 = torch.softmax(torch.tensor([4.0, 3.0]), dim=0) * scaling_factor
    assert torch.allclose(probs[0, 3], expected_probs_0[0], atol=1e-5)
    assert torch.allclose(probs[0, 2], expected_probs_0[1], atol=1e-5)
    assert probs[0, 0] == 0.0
    assert probs[0, 1] == 0.0

    # Verify routing_map for first token
    assert routing_map[0, 2].item()  # expert 2 selected
    assert routing_map[0, 3].item()  # expert 3 selected
    assert not routing_map[0, 0].item()
    assert not routing_map[0, 1].item()


def test_fused_compute_score_for_moe_aux_loss():
    """Test fused_compute_score_for_moe_aux_loss."""
    num_tokens = 4
    num_experts = 4
    topk = 2

    # Create predictable logits
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Test with softmax
    routing_map, scores = fused_compute_score_for_moe_aux_loss(
        logits=logits,
        topk=topk,
        score_function="softmax",
    )

    # Check shapes
    assert routing_map.shape == (num_tokens, num_experts)
    assert scores.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Check that scores are in fp32
    assert scores.dtype == torch.float32

    # Verify specific values for first token
    # First token: logits = [1.0, 2.0, 3.0, 4.0]
    # Softmax scores: softmax([1.0, 2.0, 3.0, 4.0])
    expected_scores_0 = torch.softmax(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)
    assert torch.allclose(scores[0], expected_scores_0, atol=1e-5)

    # Verify routing_map for first token (topk=2 should select indices [3, 2])
    assert routing_map[0, 3].item()
    assert routing_map[0, 2].item()
    assert not routing_map[0, 0].item()
    assert not routing_map[0, 1].item()


def test_fused_moe_aux_loss():
    """Test fused_moe_aux_loss."""
    num_tokens = 4
    num_experts = 4
    topk = 2

    # Create predictable probs
    probs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
            [0.1, 0.4, 0.2, 0.3],
            [0.3, 0.1, 0.4, 0.2],
        ],
        dtype=torch.float32,
    )

    # Create tokens_per_expert
    tokens_per_expert = torch.tensor([2, 3, 4, 3], dtype=torch.float32)

    # Compute aux loss
    aux_loss = fused_moe_aux_loss(
        probs=probs,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        coeff=0.01,
    )

    # Check that aux_loss is a scalar
    assert aux_loss.dim() == 0

    # Check that aux_loss is non-negative
    assert aux_loss >= 0

    # Manually compute expected aux loss
    # loss = E * coeff / (topk * T^2) * sum(aggregated_probs * tokens_per_expert)
    aggregated_probs = probs.sum(dim=0)  # [0.9, 1.0, 1.1, 1.0]
    expected_loss = torch.sum(aggregated_probs * tokens_per_expert) * (
        num_experts * 0.01 / (topk * num_tokens * num_tokens)
    )
    assert torch.allclose(aux_loss, expected_loss, atol=1e-6)


def test_fused_topk_with_score_function_group_limited():
    """Test fused_topk_with_score_function with group-limited routing."""
    num_tokens = 4
    num_experts = 8  # 8 experts divided into 2 groups
    topk = 2
    num_groups = 2  # 2 groups: [0-3] and [4-7]
    group_topk = 1  # select 1 group

    # Create predictable logits
    # Design: make group 1 (experts 4-7) have higher scores for first 2 tokens
    # and group 0 (experts 0-3) have higher scores for last 2 tokens
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 10.0, 9.0, 8.0, 7.0],  # Group 1 should be selected
            [4.0, 3.0, 2.0, 1.0, 7.0, 8.0, 9.0, 10.0],  # Group 1 should be selected
            [10.0, 9.0, 8.0, 7.0, 1.0, 2.0, 3.0, 4.0],  # Group 0 should be selected
            [7.0, 8.0, 9.0, 10.0, 4.0, 3.0, 2.0, 1.0],  # Group 0 should be selected
        ],
        dtype=torch.float32,
    )

    # Test with group-limited routing
    probs, routing_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
    )

    # Check shapes
    assert probs.shape == (num_tokens, num_experts)
    assert routing_map.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Verify group selection for first token
    # First token: group 0 scores = [1,2,3,4], group 1 scores = [10,9,8,7]
    # Group scores (sum of top-2 in each group):
    #   Group 0: top-2 are [4,3], sum = 7
    #   Group 1: top-2 are [10,9], sum = 19
    # So group 1 should be selected, then top-2 from group 1: [10, 9] (experts 4, 5)
    selected_experts_token0 = torch.where(routing_map[0])[0]
    assert all(e >= 4 for e in selected_experts_token0), (
        f"Token 0 should select from group 1, got experts {selected_experts_token0.tolist()}"
    )

    # Verify group selection for second token
    # Second token: group 0 scores = [4,3,2,1], group 1 scores = [7,8,9,10]
    # Group scores:
    #   Group 0: top-2 are [4,3], sum = 7
    #   Group 1: top-2 are [10,9], sum = 19
    # So group 1 should be selected
    selected_experts_token1 = torch.where(routing_map[1])[0]
    assert all(e >= 4 for e in selected_experts_token1), (
        f"Token 1 should select from group 1, got experts {selected_experts_token1.tolist()}"
    )

    # Verify group selection for third token
    # Third token: group 0 scores = [10,9,8,7], group 1 scores = [1,2,3,4]
    # Group scores:
    #   Group 0: top-2 are [10,9], sum = 19
    #   Group 1: top-2 are [4,3], sum = 7
    # So group 0 should be selected, then top-2 from group 0: [10, 9] (experts 0, 1)
    selected_experts_token2 = torch.where(routing_map[2])[0]
    assert all(e < 4 for e in selected_experts_token2), (
        f"Token 2 should select from group 0, got experts {selected_experts_token2.tolist()}"
    )

    # Verify group selection for fourth token
    # Fourth token: group 0 scores = [7,8,9,10], group 1 scores = [4,3,2,1]
    # Group scores:
    #   Group 0: top-2 are [10,9], sum = 19
    #   Group 1: top-2 are [4,3], sum = 7
    # So group 0 should be selected
    selected_experts_token3 = torch.where(routing_map[3])[0]
    assert all(e < 4 for e in selected_experts_token3), (
        f"Token 3 should select from group 0, got experts {selected_experts_token3.tolist()}"
    )

    # Verify specific values for first token
    # Selected logits from group 1: [10.0, 9.0]
    # Softmax: exp(10)/(exp(10)+exp(9)) ≈ 0.731, exp(9)/(exp(10)+exp(9)) ≈ 0.269
    expected_probs_0 = torch.softmax(torch.tensor([10.0, 9.0]), dim=0)
    # Check that the probabilities match (order might vary)
    probs_token0 = probs[0, routing_map[0]]
    assert torch.allclose(probs_token0.sort()[0], expected_probs_0.sort()[0], atol=1e-5)


def test_fused_topk_with_score_function_with_expert_bias():
    """Test fused_topk_with_score_function with expert bias."""
    num_tokens = 4
    num_experts = 4
    topk = 2

    # Create predictable logits for easy verification
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Create expert bias that will change the selection
    # With this bias, expert 0 and 1 will have higher scores
    expert_bias = torch.tensor([5.0, 4.0, 0.0, 0.0], dtype=torch.float32)

    # Test with expert bias
    probs, routing_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="sigmoid",
        expert_bias=expert_bias,
    )

    # Check shapes
    assert probs.shape == (num_tokens, num_experts)
    assert routing_map.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Verify specific values for first token with expert bias
    # First token: logits = [1.0, 2.0, 3.0, 4.0]
    # Sigmoid scores: [sigmoid(1), sigmoid(2), sigmoid(3), sigmoid(4)]
    sigmoid_scores = torch.sigmoid(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # Add expert bias for routing decision
    scores_for_routing = sigmoid_scores + expert_bias
    # topk=2 should select indices with highest (sigmoid_score + bias)
    # scores_for_routing = [sigmoid(1)+5, sigmoid(2)+4, sigmoid(3)+0, sigmoid(4)+0]
    #                     ≈ [0.731+5, 0.881+4, 0.953+0, 0.982+0]
    #                     ≈ [5.731, 4.881, 0.953, 0.982]
    # So experts 0 and 1 should be selected
    _, expected_top_indices = torch.topk(scores_for_routing, k=topk)
    selected_scores = sigmoid_scores[expected_top_indices]
    expected_probs_0 = selected_scores / selected_scores.sum()

    # Verify the selected experts
    selected_experts = torch.where(routing_map[0])[0]
    assert set(selected_experts.tolist()) == set(expected_top_indices.tolist())

    # Verify the probabilities
    probs_token0 = probs[0, routing_map[0]]
    assert torch.allclose(probs_token0.sort()[0], expected_probs_0.sort()[0], atol=1e-5)


def test_fused_topk_with_score_function_pre_softmax():
    """Test fused_topk_with_score_function with pre-softmax."""
    num_tokens = 4
    num_experts = 4
    topk = 2

    # Create predictable logits for easy verification
    logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 4.0, 2.0, 3.0],
            [3.0, 1.0, 4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    # Test with use_pre_softmax=True
    # This means: softmax first, then topk
    probs, routing_map = fused_topk_with_score_function(
        logits=logits,
        topk=topk,
        use_pre_softmax=True,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
    )

    # Check shapes
    assert probs.shape == (num_tokens, num_experts)
    assert routing_map.shape == (num_tokens, num_experts)

    # Check that routing_map is boolean
    assert routing_map.dtype == torch.bool

    # Check that exactly topk experts are selected per token
    assert (routing_map.sum(dim=1) == topk).all()

    # Verify specific values for first token with pre-softmax
    # First token: logits = [1.0, 2.0, 3.0, 4.0]
    # With use_pre_softmax=True: softmax first, then topk
    # Softmax scores: softmax([1.0, 2.0, 3.0, 4.0])
    softmax_scores = torch.softmax(torch.tensor([1.0, 2.0, 3.0, 4.0]), dim=0)
    # topk=2 should select indices [3, 2] (largest softmax values)
    _, expected_top_indices = torch.topk(softmax_scores, k=topk)
    expected_probs_0 = softmax_scores[expected_top_indices]

    # Verify the selected experts
    assert routing_map[0, 3].item()  # expert 3 selected
    assert routing_map[0, 2].item()  # expert 2 selected
    assert not routing_map[0, 0].item()
    assert not routing_map[0, 1].item()

    # Verify the probabilities (they should be the softmax values)
    assert torch.allclose(probs[0, 3], expected_probs_0[0], atol=1e-5)
    assert torch.allclose(probs[0, 2], expected_probs_0[1], atol=1e-5)
    assert probs[0, 0] == 0.0
    assert probs[0, 1] == 0.0


@pytest.mark.parametrize("score_function", ["sigmoid", "sqrtsoftplus"])
def test_compute_score_normalizes_non_softmax_scores(score_function):
    """Non-softmax score functions must normalize full expert scores before top-k."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)

    routing_map, scores = fused_compute_score_for_moe_aux_loss(
        logits=logits,
        topk=2,
        score_function=score_function,
    )

    raw_scores = (
        torch.sigmoid(logits)
        if score_function == "sigmoid"
        else torch.nn.functional.softplus(logits).sqrt()
    )
    expected_scores = raw_scores / raw_scores.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(scores, expected_scores)
    assert routing_map[0].sum().item() == 2
    assert routing_map[0, 2:].all().item()


@pytest.mark.parametrize(
    "function, kwargs",
    [
        (
            fused_topk_with_score_function,
            dict(
                logits=torch.ones(1, 2),
                topk=1,
                use_pre_softmax=False,
                num_groups=None,
                group_topk=None,
                scaling_factor=None,
                score_function="invalid",
                expert_bias=None,
            ),
        ),
        (
            fused_compute_score_for_moe_aux_loss,
            dict(
                logits=torch.ones(1, 2),
                topk=1,
                score_function="invalid",
            ),
        ),
    ],
)
def test_router_rejects_unknown_score_function(function, kwargs):
    with pytest.raises(ValueError, match="Invalid score_function"):
        function(**kwargs)


def test_deterministic_router_matches_selected_softmax_scores():
    """Deterministic mode uses index_put while preserving the routing result."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        probs, routing_map = fused_topk_with_score_function(
            logits=logits,
            topk=2,
            use_pre_softmax=False,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            score_function="softmax",
            expert_bias=None,
        )
    finally:
        torch.use_deterministic_algorithms(was_deterministic)

    expected = torch.zeros_like(logits)
    expected[0, 2:] = torch.softmax(logits[0, 2:], dim=0)
    torch.testing.assert_close(probs, expected)
    assert torch.equal(routing_map, expected.bool())
