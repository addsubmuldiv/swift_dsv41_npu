# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""FP8 quantization utilities for TransformerEngine Ascend PyTorch API"""

import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import torch

from transformer_engine.common.recipe import (
    DelayedScaling,
    MXFP4BlockScaling,
    MXFP8BlockScaling,
    Recipe,
    W4A16BlockScaling,
    W4A8BlockScaling,
)

from ..constants import dist_group_type
from ..utils import triton_available
from .utils import (
    check_fp8_block_scaling_support,
    check_fp8_support,
    check_mxfp4_support,
    check_mxfp8_support,
    check_nvfp4_support,
    check_recipe_support,
    check_w4a16_support,
    check_w4a8_support,
    get_default_recipe,
    get_fp8_max,
)


@dataclass(slots=True)
class FP8GlobalState:
    """Mutable process-global FP8 state stored on an instance.

    Using an instance avoids class-level `setattr(type, ...)` writes, which
    `torch.compile` cannot trace in fullgraph mode.
    """

    fp8_enabled: bool = False
    fp8_calibration: bool = False
    fp8_recipe: Optional[Recipe] = None
    fp8_distributed_group: Optional[dist_group_type] = None
    fp8_parameters: bool = False
    high_precision_init_val: bool = False
    is_first_fp8_module: bool = False
    fp8_graph_capturing: bool = False
    autocast_depth: int = 0
    global_amax_buffer: Dict[str, list] = field(default_factory=dict)
    global_amax_history_buffer: Dict[str, list] = field(default_factory=dict)
    global_scale_buffer: Dict[str, list] = field(default_factory=dict)
    hif8_state_buffer: Dict[str, List[Any]] = field(default_factory=dict)
    hif8_amax_buffer: Dict[str, torch.Tensor] = field(default_factory=dict)
    hif8_amax_buffer_layout: Dict[str, List[Optional[Tuple[int, int]]]] = field(
        default_factory=dict
    )
    hif8_amax_buffer_numel: Dict[str, int] = field(default_factory=dict)
    fp8_tensors_recompute_buffer: list[deque] = field(default_factory=list)
    autocast_arguments: Dict[Any, Tuple[Recipe, Optional[dist_group_type]]] = field(
        default_factory=dict
    )
    skip_fp8_weight_update_tensor: Optional[torch.Tensor] = None


class FP8GlobalStateManager:
    """Class to keep track of and manipulate the global
    FP8 state at different stages of execution.
    """

    quantization_state = FP8GlobalState()

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        cls.quantization_state = FP8GlobalState()

    @classmethod
    def is_fp8_available(cls) -> Tuple[bool, str]:
        """Return if fp8 support is available"""
        return check_fp8_support()

    @classmethod
    def is_mxfp8_available(cls) -> Tuple[bool, str]:
        """Return if MXFP8/current scaling support is available."""
        return check_mxfp8_support()

    @classmethod
    def is_mxfp4_available(cls) -> Tuple[bool, str]:
        """Return if MXFP4 support is available."""
        return check_mxfp4_support()

    @classmethod
    def is_fp8_block_scaling_available(cls) -> Tuple[bool, str]:
        """Return if Float8 block scaling support is available."""
        return check_fp8_block_scaling_support()

    @classmethod
    def is_nvfp4_available(cls) -> Tuple[bool, str]:
        """Return if NVFP4 support is available."""
        return check_nvfp4_support()

    @classmethod
    def is_w4a8_available(cls) -> Tuple[bool, str]:
        """Return if W4A8 support is available."""
        return check_w4a8_support()

    @classmethod
    def is_w4a16_available(cls) -> Tuple[bool, str]:
        """Return if W4A16 QAT support is available."""
        return check_w4a16_support()

    @staticmethod
    def get_meta_tensor_key(forward: bool = True) -> str:
        """Returns scaling key in `fp8_meta`."""
        if forward:
            return "scaling_fwd"
        return "scaling_bwd"

    @staticmethod
    def get_fwd_bwd_key(forward: bool = True) -> str:
        """Convert bool `forward` to string."""
        return "forward" if forward else "backward"

    @classmethod
    def get_buffer_info(cls) -> Literal["buffer_index_and_autocast_key"]:
        """
        Returns a key for `fp8_meta` that stores the module's index
        in the global buffers along with autocast information.
        """
        return "buffer_index_and_autocast_key"

    @classmethod
    def get_hif8_buffer_info(cls) -> Literal["hif8_buffer_index_and_autocast_key"]:
        """Return the fp8_meta key used by the independent HiF8 registry."""
        return "hif8_buffer_index_and_autocast_key"

    @classmethod
    def get_key_in_buffer(
        cls,
        forward: bool,
        fp8_recipe: Recipe,
        fp8_group: dist_group_type,
    ) -> str:
        """Returns a key into the global FP8 buffers."""
        autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
        fwd_bwd_key = cls.get_fwd_bwd_key(forward)
        return f"{fwd_bwd_key}_{autocast_key}"

    @classmethod
    def split_key_in_buffer(cls, key: str) -> Tuple[bool, str]:
        """Splits buffer key into relevant parts."""
        forward, autocast_key = key.split("_", 1)
        forward = forward == "forward"
        return forward, autocast_key

    @classmethod
    def _has_delayed_scaling_state(cls, fp8_meta: Dict[str, Any]) -> bool:
        """Check if fp8_meta has delayed scaling state, including CustomRecipe with DS."""
        if fp8_meta["recipe"].delayed():
            return True
        if fp8_meta["recipe"].custom():
            from .state.custom_state import CustomRecipeState

            for key in ("scaling_fwd", "scaling_bwd"):
                state = fp8_meta.get(key)
                if isinstance(state, CustomRecipeState) and state._has_delayed_scaling:
                    return True
        return False

    @classmethod
    def add_fp8_tensors_to_global_buffer(
        cls,
        fp8_meta: Dict[str, Any],
    ) -> None:
        """
        Delayed scaling only.

        The amax reduction process happens completely outside the FP8 modules.
        To participate in the reduction, the only role played by a module is
        to call this function in order to append it's FP8 tensor into a global
        buffer. There are 5 global buffers maintained, one each for amax, amax
        history, scale, scale-inverse, and non-weight-mask. Each buffer has
        keys that hold FP8 tensors. Keys have a `forward_` or `backward_` prefix
        to indicate the type of FP8 tensor, since the forward and backward
        reductions happen separately.

        Note: For CG capture, this method is called from the graphed
        wrapper. For non CG case, it's called from within the module.
        """
        # delayed scaling only function, noop for any other recipe
        if not cls._has_delayed_scaling_state(fp8_meta):
            return

        # Every module must call this function exactly once since
        # the amax tensors are static. Ensures that compatibility
        # with non-graphed modules is maintained.
        index_in_buffer = cls.get_buffer_info()  # Same index for fwd/bwd fp8 tensors.
        if index_in_buffer in fp8_meta:
            return

        fp8_meta[index_in_buffer] = []
        qstate = cls.quantization_state
        for forward in (True, False):
            fp8_meta_tensor_key = cls.get_meta_tensor_key(forward=forward)
            if fp8_meta_tensor_key not in fp8_meta:
                # Handles non-parameter FP8 modules, e.g. DPA.
                continue

            state = fp8_meta[fp8_meta_tensor_key]

            # For CustomRecipe with DS, use the inner delayed scaling recipe for buffer keys.
            if fp8_meta["recipe"].custom() and getattr(state, "_has_delayed_scaling", False):
                inner_recipe = state._inner_delayed_scaling_recipe
                key = cls.get_key_in_buffer(forward, inner_recipe, fp8_meta["fp8_group"])
                autocast_key = cls.get_unique_autocast_key(inner_recipe, fp8_meta["fp8_group"])
                qstate.autocast_arguments[autocast_key] = (inner_recipe, fp8_meta["fp8_group"])
            else:
                key = cls.get_key_in_buffer(forward, fp8_meta["recipe"], fp8_meta["fp8_group"])

            if key not in qstate.global_amax_buffer:
                qstate.global_amax_buffer[key] = [fp8_meta[fp8_meta_tensor_key].amax_history[0]]
                qstate.global_amax_history_buffer[key] = [
                    fp8_meta[fp8_meta_tensor_key].amax_history
                ]
                qstate.global_scale_buffer[key] = [fp8_meta[fp8_meta_tensor_key].scale]
            else:
                qstate.global_amax_buffer[key].append(fp8_meta[fp8_meta_tensor_key].amax_history[0])
                qstate.global_amax_history_buffer[key].append(
                    fp8_meta[fp8_meta_tensor_key].amax_history
                )
                qstate.global_scale_buffer[key].append(fp8_meta[fp8_meta_tensor_key].scale)
            fp8_meta[index_in_buffer].append(len(qstate.global_amax_buffer[key]) - 1)
            fp8_meta[index_in_buffer].append(key)

    @classmethod
    def register_hif8_states(cls, fp8_meta: Dict[str, Any]) -> None:
        """Register HiF8 states without touching delayed-scaling buffers."""
        if not fp8_meta["recipe"].hif8_delayed():
            return

        qstate = cls.quantization_state
        info_key = cls.get_hif8_buffer_info()
        registrations = fp8_meta.get(info_key, {})
        recipe = fp8_meta["recipe"]
        group = fp8_meta["fp8_group"]
        registration_signature = (
            id(qstate),
            id(recipe),
            id(group),
            id(fp8_meta.get(cls.get_meta_tensor_key(forward=True))),
            id(fp8_meta.get(cls.get_meta_tensor_key(forward=False))),
        )
        if registrations.get("signature") == registration_signature:
            for forward in (True, False):
                state = fp8_meta.get(cls.get_meta_tensor_key(forward=forward))
                if state is not None and not state.tracking_enabled:
                    state.enable_tracking()
            return

        dirty_buffer_keys = set()
        for forward in (True, False):
            meta_key = cls.get_meta_tensor_key(forward=forward)
            if meta_key not in fp8_meta:
                continue
            state = fp8_meta[meta_key]
            state.enable_tracking()
            key = cls.get_key_in_buffer(forward, recipe, group)
            autocast_key = cls.get_unique_autocast_key(recipe, group)
            qstate.autocast_arguments[autocast_key] = (
                recipe,
                group,
            )

            direction = cls.get_fwd_bwd_key(forward)
            previous = registrations.get(direction)
            entries = qstate.hif8_state_buffer.setdefault(key, [])
            position = None
            if previous is not None:
                old_position, old_key, _ = previous
                if old_key == key and old_position < len(entries):
                    position = old_position
                    entries[position] = state
                else:
                    old_entries = qstate.hif8_state_buffer.get(old_key)
                    if old_entries is not None and old_position < len(old_entries):
                        old_entries[old_position] = None
                        dirty_buffer_keys.add(old_key)
            if position is None:
                position = next(
                    (i for i, registered_state in enumerate(entries) if registered_state is state),
                    None,
                )
            if position is None:
                position = len(entries)
                entries.append(state)
            registrations[direction] = (position, key, id(state))
            dirty_buffer_keys.add(key)
        for key in dirty_buffer_keys:
            cls._refresh_hif8_amax_buffer(key)
        registrations["signature"] = registration_signature
        fp8_meta[info_key] = registrations

    @classmethod
    def _refresh_hif8_amax_buffer(cls, buffer_key: str) -> None:
        """Bind registered pending amax tensors to one reusable contiguous buffer."""
        qstate = cls.quantization_state
        entries = qstate.hif8_state_buffer.get(buffer_key, [])
        layout = []
        required_numel = 0
        prototype = None
        for state in entries:
            if state is None:
                layout.append(None)
                continue
            if prototype is None:
                prototype = state.pending_amax
            start = required_numel
            required_numel += state.pending_amax.numel()
            layout.append((start, required_numel))

        if prototype is None:
            qstate.hif8_amax_buffer.pop(buffer_key, None)
            qstate.hif8_amax_buffer_layout.pop(buffer_key, None)
            qstate.hif8_amax_buffer_numel.pop(buffer_key, None)
            return

        old_buffer = qstate.hif8_amax_buffer.get(buffer_key)
        old_layout = qstate.hif8_amax_buffer_layout.get(buffer_key, [])
        layout_extends_existing = layout[: len(old_layout)] == old_layout
        can_reuse = (
            old_buffer is not None
            and old_buffer.device == prototype.device
            and old_buffer.dtype == prototype.dtype
            and old_buffer.numel() >= required_numel
            and layout_extends_existing
        )
        if can_reuse:
            buffer = old_buffer
        else:
            minimum_capacity = max(required_numel, 8)
            capacity = 1 << (minimum_capacity - 1).bit_length()
            if old_buffer is not None:
                capacity = max(capacity, old_buffer.numel())
            buffer = torch.zeros(capacity, dtype=prototype.dtype, device=prototype.device)

        for state, segment in zip(entries, layout):
            if state is None or segment is None:
                continue
            start, end = segment
            target = buffer[start:end]
            if target.data_ptr() != state.pending_amax.data_ptr():
                target.copy_(state.pending_amax)
            state.pending_amax = target

        if required_numel < buffer.numel():
            buffer[required_numel:].zero_()
        qstate.hif8_amax_buffer[buffer_key] = buffer
        qstate.hif8_amax_buffer_layout[buffer_key] = layout
        qstate.hif8_amax_buffer_numel[buffer_key] = required_numel

    @classmethod
    def flush_hif8_states(cls, forward: bool = True) -> None:
        """Reduce pending HiF8 samples and commit one outer FP8 region."""
        qstate = cls.quantization_state
        for buffer_key, registered_states in qstate.hif8_state_buffer.items():
            fwd_update, autocast_key = cls.split_key_in_buffer(buffer_key)
            if fwd_update != forward:
                continue
            states = []
            seen_states = set()
            has_samples = False
            for state in registered_states:
                if state is not None and id(state) not in seen_states and state.has_active_region():
                    states.append(state)
                    seen_states.add(id(state))
                    if state.sampled_indices():
                        has_samples = True
            if not states:
                continue

            if has_samples:
                recipe, group = qstate.autocast_arguments[autocast_key]
                if (
                    recipe.reduce_amax
                    and torch.distributed.is_initialized()
                    and torch.distributed.get_world_size(group=group) > 1
                ):
                    used_numel = qstate.hif8_amax_buffer_numel[buffer_key]
                    contiguous_amax = qstate.hif8_amax_buffer[buffer_key][:used_numel]
                    cls.reduce_tensor_across_group_op_max(contiguous_amax, group)

            for state in states:
                state.commit_region()

    @classmethod
    def reset_hif8_amax_history(
        cls,
        *,
        forward: Optional[bool] = None,
        preserve_scale: bool = True,
    ) -> None:
        """Clear registered HiF8 histories and restart current warmup."""
        seen = set()
        for buffer_key, states in cls.quantization_state.hif8_state_buffer.items():
            fwd_state, _ = cls.split_key_in_buffer(buffer_key)
            if forward is not None and fwd_state != forward:
                continue
            for state in states:
                if state is None or id(state) in seen:
                    continue
                state.reset_amax_history(preserve_scale=preserve_scale)
                seen.add(id(state))

    @classmethod
    def reset_fp8_amax_history(
        cls,
        *,
        forward: Optional[bool] = None,
        preserve_scale: bool = True,
    ) -> None:
        """Compatibility alias that resets only the independent HiF8 policy."""
        cls.reset_hif8_amax_history(
            forward=forward,
            preserve_scale=preserve_scale,
        )

    @classmethod
    def is_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.quantization_state.fp8_enabled

    @classmethod
    def is_fp8_calibration(cls) -> bool:
        """Is FP8 calibration"""
        return cls.quantization_state.fp8_calibration

    @classmethod
    def with_fp8_parameters(cls) -> bool:
        """Should the parameters be stored as FP8"""
        return cls.quantization_state.fp8_parameters

    @classmethod
    def with_high_precision_init_val(cls) -> bool:
        """Should the high precision initial values be stored with FP8 parameters"""
        return cls.quantization_state.high_precision_init_val

    @classmethod
    def fp8_graph_capturing(cls) -> bool:
        """Is CUDA graph capture under way?"""
        if torch.compiler.is_compiling():
            assert not cls.quantization_state.fp8_graph_capturing
            return False
        return cls.quantization_state.fp8_graph_capturing or torch.npu.is_current_stream_capturing()

    @classmethod
    def is_first_fp8_module(cls):
        """Returns `True` only the first time when called multiple
        times from within the same `autocast` context.
        """
        tmp = cls.quantization_state.is_first_fp8_module
        cls.quantization_state.is_first_fp8_module = False
        return tmp

    @classmethod
    def get_fp8_recipe(cls) -> Recipe:
        """Return the fp8 recipe"""
        if cls.quantization_state.fp8_recipe is not None:
            return cls.quantization_state.fp8_recipe
        return get_default_recipe()

    @classmethod
    def get_fp8_group(cls) -> Union[dist_group_type, None]:
        """Return the fp8 group for scale/amax comm"""
        return cls.quantization_state.fp8_distributed_group

    @classmethod
    def get_autocast_state(cls) -> tuple:
        """Snapshot the autocast-related fields of the quantization state."""
        qstate = cls.quantization_state
        return (
            qstate.fp8_enabled,
            qstate.fp8_calibration,
            qstate.fp8_recipe,
            qstate.fp8_distributed_group,
            qstate.is_first_fp8_module,
            qstate.fp8_graph_capturing,
        )

    @classmethod
    def set_autocast_state(cls, state: tuple) -> None:
        """Restore a previously saved autocast state snapshot."""
        qstate = cls.quantization_state
        (
            qstate.fp8_enabled,
            qstate.fp8_calibration,
            qstate.fp8_recipe,
            qstate.fp8_distributed_group,
            qstate.is_first_fp8_module,
            qstate.fp8_graph_capturing,
        ) = state

    @staticmethod
    def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group: dist_group_type) -> None:
        """Reduce tensor across given group."""
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
                async_op=False,
            )

    @classmethod
    def reduce_and_update_fp8_tensors(
        cls,
        forward: bool = True,
    ) -> None:
        """Run the standard delayed update, then finalize independent HiF8 state."""
        # global_amax_buffer should only be non-empty for fp8 delayed scaling
        qstate = cls.quantization_state
        for (
            buffer_key,
            amax_buffer,
        ) in qstate.global_amax_buffer.items():
            # Check for forward or backward reduction.
            fwd_update, autocast_key = cls.split_key_in_buffer(buffer_key)
            if fwd_update != forward:
                continue
            if len(amax_buffer) == 0:
                continue

            # Retrieve autocast specific args.
            recipe, group = qstate.autocast_arguments[autocast_key]
            recipe: DelayedScaling

            if (
                recipe.reduce_amax
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(group=group) > 1
            ):
                contiguous_amax = torch.cat(amax_buffer)
                cls.reduce_tensor_across_group_op_max(contiguous_amax, group)
                split_and_copy(contiguous_amax, amax_buffer, [x.numel() for x in amax_buffer])

            fp8_max = get_fp8_max(recipe, forward)

            # Amax and scale update.
            use_fused = not (
                not triton_available()
                or callable(recipe.amax_compute_algo)
                or callable(recipe.scaling_factor_compute_algo)
            )

            if use_fused:
                from transformer_engine.pytorch.triton.amax_scale_update import (
                    fused_amax_and_scale_update,
                )

                try:
                    fused_amax_and_scale_update(
                        qstate.global_amax_history_buffer[buffer_key],
                        qstate.global_scale_buffer[buffer_key],
                        fp8_max,
                        recipe,
                    )
                    continue
                except Exception as e:
                    warnings.warn(
                        f"Fused Triton amax/scale update failed "
                        f"({type(e).__name__}: {e}); falling back to unfused update.",
                    )

            for amax_history, scale in zip(
                qstate.global_amax_history_buffer[buffer_key],
                qstate.global_scale_buffer[buffer_key],
            ):
                _amax_and_scale_update(amax_history, scale, fp8_max, recipe)

        if qstate.hif8_state_buffer:
            cls.flush_hif8_states(forward=forward)

    @classmethod
    def get_unique_autocast_key(
        cls,
        recipe: Optional[Recipe] = None,
        group: Optional[dist_group_type] = None,
    ):
        """
        For FP8, each autocast can be uniquely identified by the recipe and fp8 group.
        Object identity is sufficient since autocast contexts never outlive a single
        training session.
        """
        return str((str(recipe), id(group) if group is not None else None))

    @classmethod
    def autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[Recipe] = None,
        fp8_group: Optional[dist_group_type] = None,
        _graph: bool = False,
    ) -> None:
        """Set state and tracking variables for entry into FP8 region."""
        fp8_recipe = get_default_recipe() if fp8_recipe is None else fp8_recipe
        autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
        qstate = cls.quantization_state
        qstate.autocast_arguments[autocast_key] = (
            fp8_recipe,
            fp8_group,
        )

        qstate.fp8_enabled = enabled
        qstate.fp8_calibration = calibrating
        qstate.fp8_recipe = fp8_recipe
        qstate.fp8_distributed_group = fp8_group
        qstate.fp8_graph_capturing = _graph

        if qstate.autocast_depth == 0:
            qstate.is_first_fp8_module = True
        qstate.autocast_depth += 1

        if enabled:
            fp8_available, reason_for_no_fp8 = cls.is_fp8_available()
            assert fp8_available, reason_for_no_fp8
            if isinstance(fp8_recipe, MXFP4BlockScaling):
                mxfp4_available, reason_for_no_mxfp4 = cls.is_mxfp4_available()
                assert mxfp4_available, reason_for_no_mxfp4
            if isinstance(fp8_recipe, MXFP8BlockScaling):
                mxfp8_available, reason_for_no_mxfp8 = cls.is_mxfp8_available()
                assert mxfp8_available, reason_for_no_mxfp8
            if isinstance(fp8_recipe, W4A8BlockScaling):
                w4a8_available, reason_for_no_w4a8 = cls.is_w4a8_available()
                assert w4a8_available, reason_for_no_w4a8
            if isinstance(fp8_recipe, W4A16BlockScaling):
                w4a16_available, reason_for_no_w4a16 = cls.is_w4a16_available()
                assert w4a16_available, reason_for_no_w4a16
            # if isinstance(fp8_recipe, Float8BlockScaling):
            #     fp8_block_available, reason_for_no_fp8_block = cls.is_fp8_block_scaling_available()
            #     assert fp8_block_available, reason_for_no_fp8_block

    @classmethod
    def autocast_exit(cls, enabled: bool, _graph: bool) -> None:
        """Set state and tracking variables for exit from FP8 region."""
        qstate = cls.quantization_state
        qstate.autocast_depth -= 1
        # Reduce only the non-FP8 weight modules here.
        # FP8 weight modules are reduced at the end of the optimizer
        # step after the weight amax is populated.
        if enabled and qstate.autocast_depth == 0 and not _graph and torch.is_grad_enabled():
            # delayed scaling only function, for other recipes (current scaling with any granularity),
            # this is noop for other recipes because cls.global_amax_buffer is empty list
            cls.reduce_and_update_fp8_tensors(forward=True)

    @classmethod
    def copy_forward_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Copy the scaling factors and amaxes for recompute forward phase
        to ensure both forward steps are numerically same.
        """
        if fp8_meta["recipe"].hif8_delayed():
            to_copy = fp8_meta["scaling_fwd"].snapshot_for_recompute()
        elif cls._has_delayed_scaling_state(fp8_meta):
            to_copy = [
                fp8_meta["scaling_fwd"].amax_history.clone(),
                fp8_meta["scaling_fwd"].scale.clone(),
            ]
        else:
            return

        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"

        qstate = cls.quantization_state
        if buffer_position_key in fp8_meta:
            qstate.fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].append(to_copy)
        else:
            if len(qstate.fp8_tensors_recompute_buffer) == 0:
                qstate.fp8_tensors_recompute_buffer = [deque()]
            else:
                qstate.fp8_tensors_recompute_buffer.append(deque())
            qstate.fp8_tensors_recompute_buffer[-1].append(to_copy)
            fp8_meta[buffer_position_key] = len(qstate.fp8_tensors_recompute_buffer) - 1

    @classmethod
    def get_old_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Switch to the copied scaling factors and amaxes from phase
        1 forward for indentical numerical outputs.
        """
        if fp8_meta["recipe"].hif8_delayed():
            fp8_meta["updated_hif8_state_fwd"] = fp8_meta["scaling_fwd"].snapshot_for_recompute()
            buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
            stashed_fp8_meta = cls.quantization_state.fp8_tensors_recompute_buffer[
                fp8_meta[buffer_position_key]
            ].popleft()
            fp8_meta["scaling_fwd"].load_recompute_snapshot(stashed_fp8_meta)
            return
        if not cls._has_delayed_scaling_state(fp8_meta):
            return

        # Store updated amaxes and scales from phase 1 post forward.
        fp8_meta["updated_amax_history_fwd"] = fp8_meta["scaling_fwd"].amax_history.clone()
        fp8_meta["updated_scale_fwd"] = fp8_meta["scaling_fwd"].scale.clone()

        # Retrieve stashed amaxes and scales from phase 1 pre forward.
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
        stashed_fp8_meta = cls.quantization_state.fp8_tensors_recompute_buffer[
            fp8_meta[buffer_position_key]
        ].popleft()

        # Replace amaxes and scales with stashed values for phase 2 forward
        fp8_meta["scaling_fwd"].amax_history.copy_(stashed_fp8_meta[0])
        fp8_meta["scaling_fwd"].scale.copy_(stashed_fp8_meta[1])

    @classmethod
    def restore_fp8_meta_tensors(cls, fp8_meta: Dict[str, Any]) -> None:
        """Restore latest scaling factors and amaxes after recompute forward run."""
        if fp8_meta["recipe"].hif8_delayed():
            fp8_meta["scaling_fwd"].load_recompute_snapshot(fp8_meta["updated_hif8_state_fwd"])
            return
        if not cls._has_delayed_scaling_state(fp8_meta):
            return

        fp8_meta["scaling_fwd"].amax_history.copy_(fp8_meta["updated_amax_history_fwd"])
        fp8_meta["scaling_fwd"].scale.copy_(fp8_meta["updated_scale_fwd"])

    # ---------- 补弃用的方法 ----------
    get_fp8_autocast_state = get_autocast_state
    set_fp8_autocast_state = set_autocast_state
    fp8_autocast_enter = autocast_enter


def reset_hif8_amax_history(
    *,
    forward: Optional[bool] = None,
    preserve_scale: bool = True,
) -> None:
    """Clear HiF8 amax history and restart its current-scaling warmup."""
    FP8GlobalStateManager.reset_hif8_amax_history(
        forward=forward,
        preserve_scale=preserve_scale,
    )


def reset_fp8_amax_history(
    *,
    forward: Optional[bool] = None,
    preserve_scale: bool = True,
) -> None:
    """Compatibility alias for :func:`reset_hif8_amax_history`."""
    reset_hif8_amax_history(
        forward=forward,
        preserve_scale=preserve_scale,
    )


@contextmanager
def fp8_model_init(
    enabled: bool = True,
    recipe: Optional[Recipe] = None,
    preserve_high_precision_init_val: bool = False,
) -> None:
    """
    .. warning::

       fp8_model_init is deprecated and will be removed in a future release. Use
       ``quantized_model_init(enabled=..., recipe=..., preserve_high_precision_init_val=...)`` instead.

    """
    warnings.warn(
        "fp8_model_init is deprecated and will be removed in a future release. "
        "Use quantized_model_init("
        "enabled=..., recipe=..., preserve_high_precision_init_val=...) instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    # Call new implementation.
    with quantized_model_init(
        enabled=enabled,
        recipe=recipe,
        preserve_high_precision_init_val=preserve_high_precision_init_val,
    ):
        yield


@contextmanager
def quantized_model_init(
    enabled: bool = True,
    recipe: Optional[Recipe] = None,
    preserve_high_precision_init_val: bool = False,
) -> ContextManager:
    """
    Context manager for initialization of quantized parameters.

    Example usage:

    .. code-block:: python

        with quantized_model_init(enabled=True):
            model = transformer_engine.pytorch.Linear(768, 768)

        # Preserving high precision initial value to initialize master weight
        with quantized_model_init(enabled=True, preserve_high_precision_init_val=True):
            model = transformer_engine.pytorch.Linear(768, 768)
        master_weight = model.weight.get_high_precision_init_val()
        model.weight.clear_high_precision_init_val()

    Parameters
    ----------
    enabled : bool, default = True
             when enabled, Transformer Engine modules created inside this `quantized_model_init`
             region will hold only quantized copies of its parameters, as opposed to the default
             behavior where both higher precision and quantized copies are present. Setting this
             option to `True` may result in lower memory consumption and is especially
             useful for scenarios like:

             * full model training using optimizer with master weights, where the high
               precision copies of weights are already present in the optimizer.
             * inference, where only the quantized copies of the parameters are used.
             * LoRA-like fine-tuning, where the main parameters of the model do not change.
    recipe : transformer_engine.common.recipe.Recipe, default = None
            Recipe used to create the parameters. If left to None, it uses the default recipe.
    preserve_high_precision_init_val : bool, default = False
             when enabled, store the high precision tensor used to initialize quantized parameters
             in CPU memory, and add two function attributes named `get_high_precision_init_val()`
             and `clear_high_precision_init_val()` to quantized parameters to get/clear this high
             precision tensor. The purpose is that users can use this high-precision copy
             to initialize master weights, avoiding the loss of precision that can occur when
             using quantized parameters directly. Note that after the master weights are initialized,
             users should call `clear_high_precision_init_val()` to release this CPU memory.

             This functionality is *EXPERIMENTAL*.
    """
    qstate = FP8GlobalStateManager.quantization_state
    _fp8_parameters = qstate.fp8_parameters
    _fp8_recipe = qstate.fp8_recipe
    _high_precision_init_val = qstate.high_precision_init_val
    qstate.fp8_parameters = enabled
    qstate.fp8_recipe = get_default_recipe() if recipe is None else recipe
    qstate.high_precision_init_val = preserve_high_precision_init_val
    try:
        yield
    finally:
        qstate.fp8_parameters = _fp8_parameters
        qstate.fp8_recipe = _fp8_recipe
        qstate.high_precision_init_val = _high_precision_init_val


@contextmanager
def fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
) -> ContextManager:
    """
    .. warning::

       ``fp8_autocast`` is deprecated and will be removed in a future release.
       Use ``autocast(enabled=..., calibrating=..., recipe=..., group=..., _graph=...)`` instead.

    """
    warnings.warn(
        "fp8_autocast is deprecated and will be removed in a future release. "
        "Use autocast(enabled=..., calibrating=..., recipe=..., group=..., _graph=...) instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    # Call new implementation.
    with autocast(
        enabled=enabled,
        calibrating=calibrating,
        recipe=fp8_recipe,
        amax_reduction_group=fp8_group,
        _graph=_graph,
    ):
        yield


class autocast:
    """
    Context manager for quantization schemes like FP8 or FP4.

    .. code-block:: python

        with autocast(enabled=True):
            out = model(inp)

    .. note::

        Support for FP8 in the Linear layer of Transformer Engine is currently limited to tensors
        with shapes where both dimensions are divisible by 16. In terms of the input to the full
        Transformer network, this typically requires padding sequence length to be multiple of 16.

    .. note::

        With the standard :class:`DelayedScaling` recipe and
        :attr:`recipe.reduce_amax==True`, a module must not be invoked more than once inside
        one `autocast` region because later calls overwrite its pending amax. The independent
        :class:`HIF8DelayedScaling` policy instead accumulates the maximum amax across repeated
        calls and advances its cadence once at region exit.

    Parameters
    ----------
    enabled : bool, default = True
             whether or not to enable low precision quantization (FP8/FP4).
    calibrating : bool, default = False
                 calibration mode allows collecting statistics such as amax and scale
                 data of quantized tensors even when executing without quantization enabled.
                 This is useful for saving an inference ready checkpoint while training
                 using a higher precision.
    recipe : recipe.Recipe, default = None
            recipe used for low precision quantization.
    amax_reduction_group : torch._C._distributed_c10d.ProcessGroup, default = None
                          distributed group over which amaxes for the quantized tensors
                          are reduced at the end of each training step.
    """

    # Class-based context manager (instead of ``@contextmanager`` from contextlib)
    # to avoid overheads.
    __slots__ = (
        "_enabled",
        "_calibrating",
        "_recipe",
        "_amax_reduction_group",
        "_graph",
        "_fp8_state",
    )

    def __init__(
        self,
        enabled: bool = True,
        calibrating: bool = False,
        recipe: Optional["Recipe"] = None,
        amax_reduction_group: Optional["dist_group_type"] = None,
        _graph: bool = False,
    ) -> None:
        self._enabled = enabled
        self._calibrating = calibrating
        self._recipe = recipe
        self._amax_reduction_group = amax_reduction_group
        self._graph = _graph
        self._fp8_state = None

    def __enter__(self) -> "autocast":
        # Disallow nested re-entry of the same instance.
        if self._fp8_state is not None:
            raise RuntimeError(
                "autocast context manager cannot be entered more than once concurrently"
            )
        if self._enabled:
            check_recipe_support(self._recipe)
        # Save current state so we always restore it on exit.
        self._fp8_state = FP8GlobalStateManager.get_autocast_state()
        FP8GlobalStateManager.autocast_enter(
            enabled=self._enabled,
            calibrating=self._calibrating,
            fp8_recipe=self._recipe,
            fp8_group=self._amax_reduction_group,
            _graph=self._graph,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            FP8GlobalStateManager.set_autocast_state(self._fp8_state)
            FP8GlobalStateManager.autocast_exit(self._enabled, _graph=self._graph)
        finally:
            # Clear the saved state so the instance can be entered again
            # sequentially (and so a failure inside the restore path does not
            # permanently mark the instance as "active").
            self._fp8_state = None


def _update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        new_amax_history = torch.roll(amax_history, -1, 0)
        amax_history.copy_(new_amax_history)
    amax_history[0].fill_(0.0)
    return amax_history


# @torch.jit.script
def _default_get_amax_and_update_history(
    amax_history: torch.Tensor,
    amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max":
        amax = torch.max(amax_history, dim=0).values
    elif amax_compute_algo == "mean":
        amax = torch.mean(amax_history, dim=0)
    else:  # amax_compute_algo == "most_recent"
        amax = amax_history[0].clone()

    amax_history = _update_amax_history(amax_history)
    return amax_history, amax


# @jit_fuser
def _default_sf_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
    _fp32_max: float = torch.finfo(torch.float32).max,  # finfo not available in jitter
) -> torch.Tensor:
    """Default function to convert amax to scaling factor.
    Computing the scaling factor requires consideration of the following scenarios:
    1. amax == 0:
       No action is possible, set scale to the previous scale (or 1).
    2. 0 < amax < tiny_amax
       The amax is too tiny that the scale becomes infinite in FP32.
       Set scale = FP32_max
    3. tiny_amax <= amax < FP32_max:
       Set scale = FP8_max (or scaled_max) / amax
    4. When amax == inf or amax == nan:
       No action is possible, set scale to the previous scale (or 1).
    """
    sf = (fp8_max / amax) / (2**margin)
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(torch.isinf(sf), torch.full_like(sf, _fp32_max), sf)
    scale.copy_(sf)
    return scale


def _compute_amax_and_update_history(
    amax_history: torch.Tensor,
    amax_compute_algo: Union[Callable, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the amax from the history."""
    if callable(amax_compute_algo):
        amax = amax_compute_algo(amax_history)
        amax_history = _update_amax_history(amax_history)
        return amax_history, amax
    return _default_get_amax_and_update_history(
        amax_history,
        amax_compute_algo,
    )


def _compute_scaling_factor(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> torch.Tensor:
    """Convert amax to scaling factor."""
    if recipe.scaling_factor_compute_algo is None:
        return _default_sf_compute(
            amax,
            scale,
            fp8_max,
            recipe.margin,
        )
    return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def _amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> None:
    """Updates FP8 meta tensors."""
    new_amax_history, amax = _compute_amax_and_update_history(
        amax_history,
        recipe.amax_compute_algo,
    )
    new_scale = _compute_scaling_factor(amax, scale, fp8_max, recipe)
    scale.copy_(new_scale)
    amax_history.copy_(new_amax_history)


def split_and_copy(
    buffer: torch.Tensor,
    outputs: List[torch.Tensor],
    chunk_sizes: List[int],
) -> None:
    """Split `buffer` by `chunk_sizes` and copy into `outputs`."""
    splits = buffer.split(chunk_sizes)
    torch._foreach_copy_(outputs, splits)
