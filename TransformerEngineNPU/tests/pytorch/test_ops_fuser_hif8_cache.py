# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""HiF8 fusion-plan caching tests without tensor execution or NPU allocation."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from transformer_engine.common.recipe import HIF8DelayedScaling
from transformer_engine.pytorch.ops.fuser import OperationFuser


class TestHIF8FusionCache(unittest.TestCase):
    """Exercise fusion selection independently of device-specific operations."""

    def setUp(self):
        self.op = SimpleNamespace(
            is_fused_op=False,
            num_extra_inputs=0,
            num_extra_outputs=0,
            _extra_input_channels=[],
            _extra_output_channels=[],
            _extra_output_to_caller=[],
            _lock_extra_tensor_channels=Mock(),
            parameters=lambda: [],
            reset_recipe_state=Mock(),
            pre_first_fuser_forward=Mock(),
        )
        self.fusion = Mock(side_effect=lambda ops, *, recipe: ops)
        for name, functions in (
            ("forward_backward_fusion_functions", [self.fusion]),
            ("forward_fusion_functions", []),
            ("backward_fusion_functions", []),
        ):
            patcher = patch.object(OperationFuser, name, functions)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.fuser = OperationFuser([self.op])

    def fuse(self, recipe, grad=True):
        self.fuser.maybe_fuse_ops(grad, recipe, SimpleNamespace(requires_grad=True), [[]])
        return self.fuser._forward_ops, self.fuser._backward_ops

    def test_unhashable_recipe_reuses_both_recompute_plans(self):
        recipe = HIF8DelayedScaling()
        no_grad = self.fuse(recipe, False)
        grad = self.fuse(recipe, True)
        for _ in range(3):
            for enabled, expected, boundary in ((False, no_grad, 1), (True, grad, 0)):
                actual = self.fuse(recipe, enabled)
                self.assertIs(actual[0], expected[0])
                self.assertIs(actual[1], expected[1])
                self.assertEqual(self.fuser.first_op_requiring_backward, boundary)
        self.assertEqual(self.fusion.call_count, 2)
        self.op.reset_recipe_state.assert_called_once_with(recipe=recipe)

    def test_equal_new_recipe_reuses_plan(self):
        first = self.fuse(HIF8DelayedScaling())
        second = self.fuse(HIF8DelayedScaling())
        self.assertIs(first[0], second[0])
        self.assertEqual(self.fusion.call_count, 1)
        self.assertEqual(self.op.reset_recipe_state.call_count, 1)

    def test_changed_recipe_invalidates_both_plans(self):
        recipe = HIF8DelayedScaling()
        self.fuse(recipe, False)
        self.fuse(recipe, True)
        changed = HIF8DelayedScaling(hif8_input_margin=9)
        self.fuse(changed, True)
        self.fuse(changed, False)
        self.assertEqual(self.fusion.call_count, 4)
        self.assertEqual(len(self.fuser._fused_ops_cache), 2)
        self.assertEqual(self.op.reset_recipe_state.call_count, 2)
        self.op.reset_recipe_state.assert_called_with(recipe=changed)

    def test_in_place_change_invalidates_plan(self):
        recipe = HIF8DelayedScaling()
        before = self.fuse(recipe)
        recipe.amax_history_len = 64
        after = self.fuse(recipe)
        self.assertIsNot(before[0], after[0])
        self.assertEqual(self.fusion.call_count, 2)
        self.assertEqual(self.op.reset_recipe_state.call_count, 2)

    def test_unhashable_margin_factory(self):
        class MarginFactory:
            __hash__ = None

            def __call__(self, role):
                return 2048

        recipe = HIF8DelayedScaling(margin_factory=MarginFactory())
        self.fuse(recipe)
        self.fuse(recipe)
        self.assertEqual(self.fusion.call_count, 1)

    def test_disable_and_reenable_quantization(self):
        recipe = HIF8DelayedScaling()
        self.fuse(recipe)
        self.fuse(None)
        self.fuse(recipe)
        self.assertEqual(self.op.reset_recipe_state.call_count, 3)
        self.op.pre_first_fuser_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
