# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

"""Base modules and utilities for TransformerEngine Ascend PyTorch API"""

import io
from functools import cached_property
import os
import pickle
import warnings
from contextlib import contextmanager
from enum import Enum
from types import MethodType
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import torch
import torch.nn.functional as F  # noqa: N812
from torch.distributed.tensor import DTensor

from transformer_engine.common.recipe import DelayedScaling, Recipe
from ._common import _ParameterInitMeta, noop_cat

from ..constants import dist_group_type
from ..distributed import (
    _fsdp_gather_tensors,
    gather_along_dim,
    in_fp8_activation_recompute_phase,
    is_fp8_activation_recompute_enabled,
)
from ..quantization import (
    CustomRecipeState,
    DelayedScalingRecipeState,
    Float8BlockScalingRecipeState,
    Float8CurrentScalingRecipeState,
    FP8GlobalStateManager,
    HIF8DelayedScalingRecipeState,
    MXFP4BlockScalingRecipeState,
    MXFP832x32BlockScalingRecipeState,
    MXFP8BlockScalingRecipeState,
    QuantizerRole,
    RecipeState,
    W4A16BlockScalingRecipeState,
    W4A8BlockScalingRecipeState,
)
from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage, Quantizer
from ..tensor import (
    Float8TensorStorage,
    MXFP4TensorStorage,
    MXFP8TensorStorage,
    W4A8WeightTensorStorage,
)
from ...debug.pytorch.debug_quantization import (
    DebugQuantizedTensor,
    DebugQuantizer,
    GroupedDebugQuantizedTensor,
    GroupedDebugQuantizer,
)
from ...debug.pytorch.debug_state import TEDebugState
from ...debug.pytorch.utils import any_feature_enabled, next_iter_when_debug_should_be_run

_2X_ACC_FPROP = False
_2X_ACC_DGRAD = True
_2X_ACC_WGRAD = True
_dummy_wgrads = {}
_ub_communicators = None
_MIN_STREAM_PRIORITY, _MAX_STREAM_PRIORITY = None, None
layers_atomic_ring_exchange = []

_LAYER_COUNT = 0


class UserBufferQuantizationMode(Enum):
    """UserBufferQuantizationMode is an enum that represents the quantization mode of the UserBuffer."""

    NONE = "none"
    FP8 = "fp8"


def setup_dummy_wgrad(weight, main_grad=None):
    """Return a dummy wgrad for MCore DDP fused accumulation paths."""
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    weight.grad_added_to_main_grad = True
    # MCore only needs a dummy gradient to make the post-accumulate hook run
    # synchronously when gradient reduction overlaps backward.  In the
    # non-overlap path the fused kernel has already populated ``main_grad`` and
    # final grad synchronization visits every bucket explicitly, so retaining
    # one full-sized placeholder per distinct weight shape is unnecessary.
    if not getattr(weight, "requires_dummy_wgrad", True):
        return None
    zero = getattr(weight, "zero_out_wgrad", False)
    if main_grad is None:
        main_grad = (
            weight.get_main_grad() if hasattr(weight, "__fsdp_param__") else weight.main_grad
        )
    _wgrad = get_dummy_wgrad(
        main_grad.shape,
        weight.dtype,
        zero,
        weight.device,
    )
    return _wgrad


def get_dummy_wgrad(shape: list, dtype: torch.dtype, zero=False, device=None) -> torch.Tensor:
    """Returns a dummy tensor of given shape."""
    key = (*shape, dtype, zero)
    if key not in _dummy_wgrads:
        _dummy_wgrads[key] = torch.empty(
            shape,
            dtype=dtype,
            device="npu" if device is None else device,
            requires_grad=False,
        )
        if zero:
            _dummy_wgrads[key].fill_(0)
    return _dummy_wgrads[key].detach()


def initialize_ub(
    shape: list,
    tp_size: int,
    use_fp8: bool = False,
    quantization_modes: List[UserBufferQuantizationMode] = None,
    dtype: torch.dtype = torch.bfloat16,
    ub_cfgs: Optional[Union[dict, List[dict]]] = None,
    bootstrap_backend: Union[str, torch.distributed.Backend] = None,
) -> None:
    """Initialize user buffers for communication."""
    pass


def destroy_ub() -> None:
    """Destroy user buffers."""
    global _ub_communicators
    _ub_communicators = None


def fill_userbuffers_buffer_for_all_gather(
    tensor: torch.Tensor,
    ub_comm: Any,
    ub_obj: Any,
    tp_size: int,
    tp_rank: int,
    ub_algo: Any,
) -> None:
    """Fill user buffers for all gather operation."""
    pass


def get_ub(
    shape: List[int],
    dtype: torch.dtype,
    ub_comm: Any,
    ub_algo: Any,
) -> Any:
    """Get user buffer."""
    pass


def _check_fp8_reduce_and_update():
    """Check if this is the first FP8 module (for backward reduce-and-update)."""
    reset = in_fp8_activation_recompute_phase()
    if reset:
        qstate = FP8GlobalStateManager.quantization_state
        _first_fp8_module = qstate.is_first_fp8_module
    result = FP8GlobalStateManager.is_first_fp8_module()
    if reset:
        qstate.is_first_fp8_module = _first_fp8_module
    return result


class FP8Meta(TypedDict):
    scaling_fwd: RecipeState
    scaling_bwd: RecipeState

    fp8_checkpoint: bool
    fp8_group: Optional[torch.distributed.ProcessGroup]
    recipe: Recipe
    num_gemms: int
    fp8_max_fwd: float
    fp8_max_bwd: float
    # ---------- cache ----------
    buffer_index_and_autocast_key: Tuple[int, str, int, str]
    global_fp8_buffer_pos_fwd_recompute: int


class _Quantizers(TypedDict):
    scaling_fwd: List["Quantizer"]
    scaling_bwd: List["Quantizer"]


def _is_weight_workspace_valid(
    workspace: QuantizedTensorStorage,
    quantizer: Quantizer,
) -> bool:
    """Check if a cached weight workspace is compatible with the quantizer's current usage."""
    if isinstance(workspace, Float8TensorStorage):
        if quantizer.columnwise_usage and workspace._data is None:
            return False
    elif isinstance(workspace, MXFP8TensorStorage):
        if quantizer.rowwise_usage and workspace._rowwise_data is None:
            return False
        if quantizer.columnwise_usage and workspace._columnwise_data is None:
            return False
    elif isinstance(workspace, MXFP4TensorStorage):
        if quantizer.rowwise_usage and workspace._rowwise_data is None:
            return False
        if quantizer.columnwise_usage and workspace._columnwise_data is None:
            return False
    elif isinstance(workspace, W4A8WeightTensorStorage):
        if quantizer.rowwise_usage and workspace._mxfp4_rowwise_data is None:
            return False
        if quantizer.columnwise_usage and workspace._mxfp8_columnwise_data is None:
            return False
    if isinstance(workspace, (DebugQuantizedTensor, GroupedDebugQuantizedTensor)) != isinstance(
        quantizer, (DebugQuantizer, GroupedDebugQuantizer)
    ):
        return False
    return True


def quantize_weight(
    *,
    tensor: Optional[torch.Tensor] = None,
    quantizer: Optional[Quantizer] = None,
    workspace: Optional[QuantizedTensorStorage] = None,
    update_workspace: bool = True,
    skip_update_flag: Optional[torch.Tensor] = None,
    fsdp_group: Optional["dist_group_type"] = None,
    workspace_dtype: Optional[torch.dtype] = None,
    cache: bool = False,
    group_list=None,
) -> Tuple[QuantizedTensorStorage, Optional[QuantizedTensorStorage]]:
    """Quantize a weight tensor, optionally reusing a cached workspace.

    Parameters
    ----------
    tensor: torch.Tensor, optional
        Weight tensor to quantize.
    quantizer: Quantizer, optional
        Quantizer for casting the weight.
    workspace: QuantizedTensorStorage, optional
        Previously cached workspace (from the module's ``_fp8_workspaces``).
        ``None`` indicates a cache miss.
    update_workspace: bool, default = True
        Whether to update an existing workspace with fresh values.
    skip_update_flag: torch.Tensor, optional
        GPU flag to conditionally skip the update.
    fsdp_group: dist_group_type, optional
        FSDP process group the weights are distributed over.
    workspace_dtype: torch.dtype, optional
        High-precision dtype for debug quantization workspaces.
    cache: bool, default = False
        If ``True`` and a new workspace is created, it will be returned
        as the second element so the caller can store it.

    Returns
    -------
    (weightmat, new_workspace)
        *weightmat*: quantized weight ready for GEMM.
        *new_workspace*: non-``None`` only when a workspace was created and
        ``cache=True``. The caller should store it in
        ``_fp8_workspaces``.
    """

    if quantizer is not None and quantizer.can_reuse(tensor):
        update_rowwise = True if quantizer.rowwise_usage else None
        update_columnwise = True if quantizer.columnwise_usage else None
        tensor.update_usage(
            rowwise_usage=update_rowwise,
            columnwise_usage=update_columnwise,
        )

        if isinstance(quantizer, DebugQuantizer):
            out = quantizer.wrap_quantized_tensor(tensor)
            return out, None

        return tensor, None

    if workspace is not None and quantizer is not None:
        if not _is_weight_workspace_valid(workspace, quantizer):
            workspace = None

    if (
        workspace is not None
        and tensor is not None
        and fsdp_group is not None
        and workspace.data.shape != tensor.data.shape
    ):
        _fsdp_gather_tensors(fsdp_group, [tensor.data.shape], workspace)

    if workspace is not None:
        if skip_update_flag is not None:
            update_workspace = True
        if update_workspace:
            if tensor is None:
                raise ValueError("tensor kwarg must be provided to update FP8 workspace")
            if group_list is None:
                workspace.quantize_(tensor, noop_flag=skip_update_flag)
            else:
                workspace.grouped_quantize_(tensor, group_list, noop_flag=skip_update_flag)
        return workspace, None

    if tensor is None or quantizer is None:
        raise ValueError("tensor and quantizer kwargs must be provided to construct FP8 workspace")
    if cache:
        saved_internal = quantizer.internal
        quantizer.internal = False
    if group_list is None:
        out = quantizer.quantize(tensor, dtype=workspace_dtype)
    else:
        out = quantizer.grouped_quantize(tensor, group_list, dtype=workspace_dtype)
    if cache:
        quantizer.internal = saved_internal
        return out, out
    return out, None


class TransformerEngineBaseModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.fp8_initialized = False
        self.fp8 = False
        self.fp8_calibration = False
        self.fp8_meta: FP8Meta = {"fp8_checkpoint": False, "fp8_group": None}
        self.fp8_meta_tensors_initialized = False
        self.quantizers: _Quantizers = {"scaling_fwd": [], "scaling_bwd": []}
        self.tp_group = None
        self.tp_size = 1
        self.tp_group_initialized = False
        self.sequence_parallel = False
        self.param_init_meta = {}
        self.primary_weights_in_fp8 = FP8GlobalStateManager.with_fp8_parameters()
        self.preserve_high_precision_init_val = FP8GlobalStateManager.with_high_precision_init_val()
        self.fsdp_wrapped = False
        self.fsdp_group = None
        self._fp8_workspaces: Dict[str, "QuantizedTensor"] = {}
        self.activation_dtype: Optional[torch.dtype] = None
        self.wgrad_accumulation_and_reduce_hooks = []
        self.wgrad_store = None
        self._output_quantizer_role: Optional[QuantizerRole] = None
        self._grad_input_quantizer_role: Optional[QuantizerRole] = None
        # fast set attr
        self.allow_different_data_and_param_types = False
        self.next_iter_when_debug_should_be_run = 0

        if not TEDebugState.debug_enabled:
            TEDebugState.initialize()
        self._validate_name()

    def fast_setattr(self, name: str, value: Any) -> None:
        """
        Fast version of the Module's set attribute function.
        Should be used for regular attributes, but not properties nor parameters/buffers.
        """
        self.__dict__[name] = value

    def module_setattr(self, name: str, value: Any) -> None:
        """
        Regular version of the Module's set attribute function.
        Should be used only when the fast version cannot be used - for the properties,
        parameters and buffers.
        """
        super().__setattr__(name, value)

    def is_debug_iter(self) -> bool:
        """Return whether this module should run debug quantizers this iteration."""
        debug = TEDebugState.debug_enabled
        if not debug:
            return False
        if torch.compiler.is_compiling():
            raise RuntimeError("Debug mode is not supported during model export or torch.compile.")

        iteration = TEDebugState.get_iteration()
        started_new_iteration = iteration != getattr(self, "debug_last_iteration", None)
        if started_new_iteration:
            if self.next_iter_when_debug_should_be_run is None:
                debug = False
            else:
                debug = iteration >= self.next_iter_when_debug_should_be_run
            self.fast_setattr("debug_enabled_in_this_iteration", debug)
        else:
            debug = self.debug_enabled_in_this_iteration

        self.fast_setattr("debug_last_iteration", iteration)

        if self.wgrad_store is not None and debug and self.wgrad_store.delay_wgrad_compute():
            raise RuntimeError("Delayed wgrad compute is not supported in debug mode.")

        return debug

    def no_debug_features_active(self, quantizers):
        """Return whether wrapped quantizers leave normal behavior unchanged."""
        self.fast_setattr(
            "next_iter_when_debug_should_be_run",
            next_iter_when_debug_should_be_run(quantizers),
        )
        return not any_feature_enabled(quantizers)

    def _validate_name(self):
        """Assign a stable default debug layer name when none was provided."""
        if self.name is None:
            self.name = f"Layer_{TEDebugState.get_layer_count()}"

    def _check_weight_tensor_recipe_correspondence(self) -> None:
        """
        Verify that the weight tensor types match their corresponding recipe type.
        This is invoked in the forward().

        This establishes a 1:1 correspondence between recipe types and tensor types:
        - DelayedScaling → Float8Tensor
        - Float8CurrentScaling → Float8Tensor
        - MXFP8BlockScaling → MXFP8Tensor
        - Float8BlockScaling → Float8BlockTensor

        Example case to check: recipe is DelayedScaling (DelayedScaling is set in autocast()),
        but the weight tensor is MXFP8Tensor (MXFP8BlockScaling is set in quantized_model_init()).
        """
        if not self.fp8 and not self.fp8_calibration:
            return
        if not self.primary_weights_in_fp8:
            return
        if not hasattr(self, "weight_names") or not self.weight_names:
            return

        recipe = self.fp8_meta["recipe"]
        weight_tensors = [getattr(self, name) for name in self.weight_names]
        for i, tensor in enumerate(weight_tensors):
            if isinstance(tensor, QuantizedTensorStorage):
                quantizer = tensor._get_quantizer()
                if quantizer is None:
                    continue
                compatible_recipe_class = quantizer._get_compatible_recipe()
                if compatible_recipe_class is None:
                    continue
                if not isinstance(recipe, compatible_recipe_class):
                    raise RuntimeError(
                        f"Recipe mismatch for '{self.weight_names[i]}': tensor supports recipe"
                        f" {compatible_recipe_class.__name__}, but got {recipe.__class__.__name__}."
                        " Please check the recipes assigned during quantized_model_init() and"
                        " autocast() calls."
                    )

    # cached_property: per-instance cache, freed with the module (lru_cache would leak weights)
    @cached_property
    def is_fsdp2(self) -> bool:
        """Whether this module is wrapped with FSDP2."""
        try:
            from ..distributed import _get_module_fsdp_state

            _get_module_fsdp_state(self)
        except (RuntimeError, ImportError):
            return False
        return True

    @property
    def output_quantizer_role(self) -> Optional[QuantizerRole]:
        return self._output_quantizer_role

    @output_quantizer_role.setter
    def output_quantizer_role(self, role: Optional[QuantizerRole]) -> None:
        if role == self._output_quantizer_role:
            return
        self._output_quantizer_role = role
        if self.fp8_meta_tensors_initialized:
            self.fp8_meta_tensors_initialized = False

    @property
    def grad_input_quantizer_role(self) -> Optional[QuantizerRole]:
        return self._grad_input_quantizer_role

    @grad_input_quantizer_role.setter
    def grad_input_quantizer_role(self, role: Optional[QuantizerRole]) -> None:
        if role == self._grad_input_quantizer_role:
            return
        self._grad_input_quantizer_role = role
        if self.fp8_meta_tensors_initialized:
            self.fp8_meta_tensors_initialized = False

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        return None

    def _num_quantizer_gemms(self) -> int:
        """Number of GEMMs to use for quantizer slot allocation.

        Defaults to ``self.fp8_meta["num_gemms"]``. Subclasses can override
        this to allocate a different number of quantizer slots than the
        actual number of GEMMs (e.g. when several GEMMs share a single
        quantizer group).
        """
        return self.fp8_meta["num_gemms"]

    def adjust_amax_history_length(self, length: int, fwd: Optional[bool] = None) -> None:
        """
        Delayed scaling only.

        Increase or decrease size of amax history based on given `length`.

        .. warning::
            This changes the underlying amax memory location.
        """
        if fwd is None:
            fp8_meta_tensor_keys = ("scaling_fwd", "scaling_bwd")
        else:
            fp8_meta_tensor_keys = ("scaling_fwd" if fwd else "scaling_bwd",)

        for meta_key in fp8_meta_tensor_keys:  # type: Literal["scaling_fwd", "scaling_bwd"]
            if meta_key not in self.fp8_meta:
                # Handles non-parameter FP8 modules, e.g. DPA.
                continue
            recipe_state = self.fp8_meta[meta_key]
            assert isinstance(recipe_state, DelayedScalingRecipeState)
            curr_len = recipe_state.amax_history.shape[0]
            if length == curr_len:
                continue
            if length < curr_len:
                recipe_state.amax_history = recipe_state.amax_history[:length].clone()
            elif length > curr_len:
                extra_rows = length - curr_len
                recipe_state.amax_history = F.pad(
                    recipe_state.amax_history, pad=(0, 0, 0, extra_rows)
                )

            # Update quantizers with new amax pointers.
            self.quantizers[meta_key] = recipe_state.make_quantizers()
            # Make sure weight tensors has correct quantizers
            self._update_weight_quantizers()

            # Update the global buffers with new amax and history pointers.
            if FP8GlobalStateManager.get_buffer_info() in self.fp8_meta:
                fwd_pos, fwd_key, bwd_pos, bwd_key = self.fp8_meta[
                    FP8GlobalStateManager.get_buffer_info()
                ]
                qstate = FP8GlobalStateManager.quantization_state
                if meta_key == "scaling_fwd":
                    pos, buffer_key = fwd_pos, fwd_key
                else:
                    pos, buffer_key = bwd_pos, bwd_key
                if buffer_key in qstate.global_amax_buffer:
                    if buffer_key not in qstate.global_amax_history_buffer:
                        raise RuntimeError(
                            "TE internal error during amax history change: "
                            f"buffer_key '{buffer_key}' found in global_amax_buffer "
                            "but missing from global_amax_history_buffer"
                        )
                    qstate.global_amax_history_buffer[buffer_key][pos] = recipe_state.amax_history
                    qstate.global_amax_buffer[buffer_key][pos] = recipe_state.amax_history[0]

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        fp8_meta_tensor_key: Literal["scaling_fwd", "scaling_bwd"] = (
            "scaling_fwd" if fwd else "scaling_bwd"
        )

        # Return early if recipe state matches recipe
        if self.fp8_meta_tensors_initialized:
            recipe_state = self.fp8_meta[fp8_meta_tensor_key]
            if recipe.hif8_delayed() and isinstance(recipe_state, HIF8DelayedScalingRecipeState):
                if recipe == recipe_state.recipe:
                    return
            if recipe.delayed() and isinstance(recipe_state, DelayedScalingRecipeState):
                self.adjust_amax_history_length(recipe.amax_history_len, fwd=fwd)
                return
            if recipe.mxfp8_32x32() and isinstance(recipe_state, MXFP832x32BlockScalingRecipeState):
                return
            if (
                recipe.mxfp8()
                and not recipe.mxfp8_32x32()
                and isinstance(recipe_state, MXFP8BlockScalingRecipeState)
            ):
                return
            if recipe.mxfp4() and isinstance(recipe_state, MXFP4BlockScalingRecipeState):
                return
            if recipe.w4a8() and isinstance(recipe_state, W4A8BlockScalingRecipeState):
                return
            if recipe.w4a16() and isinstance(recipe_state, W4A16BlockScalingRecipeState):
                return
            if recipe.float8_current_scaling() and isinstance(
                recipe_state, Float8CurrentScalingRecipeState
            ):
                return
            if recipe.float8_block_scaling() and isinstance(
                recipe_state, Float8BlockScalingRecipeState
            ):
                return
            if recipe.custom() and isinstance(recipe_state, CustomRecipeState):
                if recipe_state.recipe is recipe:
                    return

        # Max. number of fp8 tensors per GEMM = 3 (input, weight, output) for fwd and
        # 2 (grad_output and grad_input) for bwd
        num_fp8_tensors = (
            self._num_quantizer_gemms() * 3 if fwd else self._num_quantizer_gemms() * 2
        )

        # Initialize recipe state and quantizers
        roles = self.get_quantizer_roles(  # pylint: disable=assignment-from-none
            fwd=fwd, num_quantizers=num_fp8_tensors
        )
        if roles is None and (recipe.hif8_delayed() or recipe.mxfp8_32x32()):
            tensor_types = ("input", "weight", "output") if fwd else ("grad_output", "grad_input")
            roles = [
                QuantizerRole(tensor_type=tensor_types[index % len(tensor_types)])
                for index in range(num_fp8_tensors)
            ]
        if roles is not None:
            assert len(roles) == num_fp8_tensors, (
                f"Recipe roles must match number of quantizers ({len(roles)=} vs {num_fp8_tensors=})"
            )
        recipe_state = RecipeState.create(
            recipe,
            mode=("forward" if fwd else "backward"),
            num_quantizers=num_fp8_tensors,
            roles=roles,
        )

        old_state = self.fp8_meta.get(fp8_meta_tensor_key)
        old_amax_history = getattr(old_state, "amax_history", None)
        if old_state is not None:
            recipe_state.inherit_state_from(old_state)

        self.fp8_meta[fp8_meta_tensor_key] = recipe_state
        self.quantizers[fp8_meta_tensor_key] = recipe_state.make_quantizers()

        # If this rebuild replaced the delayed-scaling amax buffers with freshly
        # allocated tensors (i.e. they could NOT be inherited in-place), any
        # global amax/scale registration still points at the old tensors and is
        # now dangling: the reduction would keep updating dead buffers while the
        # live quantizers use the new ones. Drop the registration key so the
        # next prepare_forward re-registers the current tensors. When the old
        # buffers were inherited (same storage), the registration stays valid.
        new_amax_history = getattr(recipe_state, "amax_history", None)
        if old_amax_history is not None:
            rebound = (
                new_amax_history is not None
                and new_amax_history.data_ptr() == old_amax_history.data_ptr()
            )
            if not rebound:
                # Also covers the case where the rebuilt state no longer uses
                # delayed scaling at all: the old registration must not be
                # treated as live either way. (The orphaned entries inside the
                # global buffer lists themselves cannot be removed without
                # shifting other modules' recorded positions; they are benign
                # but wasteful, matching upstream behaviour.)
                self.fp8_meta.pop(FP8GlobalStateManager.get_buffer_info(), None)

    def _update_weight_quantizers(self) -> None:
        """Update the quantizers for the weight tensors."""
        weight_tensors = self._get_weight_tensors()
        weight_quantizers = self._get_weight_quantizers()
        if len(weight_tensors) != len(weight_quantizers):
            raise ValueError(
                f"Number of weight tensors ({len(weight_tensors)}) and quantizers "
                f"({len(weight_quantizers)}) must match"
            )
        for weight, quantizer in zip(weight_tensors, weight_quantizers):
            if quantizer is not None and isinstance(weight, QuantizedTensorStorage):
                weight.update_quantizer(quantizer)

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement _get_weight_tensors function"
        )

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement _get_weight_quantizers function"
        )

    def init_fp8_meta_tensors(self, recipe: Recipe) -> None:
        """Init scales and amaxes."""
        self.set_meta_tensor(True, recipe)
        self.set_meta_tensor(False, recipe)

        self.fast_setattr("fp8_meta_tensors_initialized", True)

    def get_fp8_meta_tensors(self) -> None:
        """Get scales and amaxes."""
        fwd_key, bwd_key = "scaling_fwd", "scaling_bwd"
        if fwd_key not in self.fp8_meta or bwd_key not in self.fp8_meta:
            return None

        fp8_meta_tensors = {fwd_key: [], bwd_key: []}
        with torch.no_grad():
            for key in (fwd_key, bwd_key):
                fp8_meta_tensors[key].append(self.fp8_meta[key].scale.clone())
                fp8_meta_tensors[key].append(self.fp8_meta[key].amax_history.clone())
        return fp8_meta_tensors

    def reset_fp8_meta_tensors(self, fp8_meta_tensors=None) -> None:
        """Reset scales and amaxes."""

        def reset(key: Literal["scaling_fwd", "scaling_bwd"]):
            if key in self.fp8_meta:
                recipe_state = self.fp8_meta[key]
                if isinstance(recipe_state, HIF8DelayedScalingRecipeState):
                    if fp8_meta_tensors is None:
                        recipe_state.reset_amax_history(preserve_scale=False)
                    else:
                        if key not in fp8_meta_tensors:
                            raise KeyError(
                                f"Cannot reset fp8 tensors: key '{key}' not found in "
                                f"fp8_meta_tensors. Available keys: "
                                f"{list(fp8_meta_tensors.keys())}"
                            )
                        recipe_state.scale.copy_(fp8_meta_tensors[key][0])
                        recipe_state.scale_inv = torch.reciprocal(recipe_state.scale)
                        recipe_state.reset_amax_history(preserve_scale=True)
                        recipe_state.amax_history.copy_(fp8_meta_tensors[key][1])
                    return
                if fp8_meta_tensors is None:
                    self.fp8_meta[key].scale.copy_(torch.ones_like(self.fp8_meta[key].scale))
                    self.fp8_meta[key].amax_history.copy_(
                        torch.zeros_like(self.fp8_meta[key].amax_history)
                    )
                else:
                    if key not in fp8_meta_tensors:
                        raise KeyError(
                            f"Cannot reset fp8 tensors: key '{key}' not found in fp8_meta_tensors. "
                            f"Available keys: {list(fp8_meta_tensors.keys())}"
                        )
                    self.fp8_meta[key].scale.copy_(fp8_meta_tensors[key][0])
                    self.fp8_meta[key].amax_history.copy_(fp8_meta_tensors[key][1])

        with torch.no_grad():
            reset("scaling_fwd")
            reset("scaling_bwd")

    def _has_delayed_scaling_state_for_checkpoint(self) -> bool:
        """Whether this module owns delayed-scaling buffers to (de)serialize.

        True for a native DelayedScaling recipe, and also for a CustomRecipe
        whose quantizer slots resolved to delayed scaling (composed
        ``DelayedScalingRecipeState`` inside ``CustomRecipeState``). In both
        cases ``scaling_fwd`` / ``scaling_bwd`` expose ``.scale`` and
        ``.amax_history`` that must be checkpointed.
        """
        recipe = self.fp8_meta.get("recipe")
        if recipe is None:
            return False
        if recipe.delayed():
            return True
        if recipe.custom():
            for key in ("scaling_fwd", "scaling_bwd"):
                state = self.fp8_meta.get(key)
                if isinstance(state, CustomRecipeState) and state._has_delayed_scaling:
                    return True
        return False

    def get_extra_state(self) -> torch.Tensor:
        """Save before checkpointing.
        # This implementation is working around a few issues:
        #
        # (1) PyTorch's "extra state" infrastructure might be able to
        #     support any picklable type, but they make no guarantees.
        #     We have experienced problems (e.g. in ONNX export) with
        #     non-tensor extra state.
        # (2) PyTorch's checkpointing infrastructure does not remap
        #     devices for "extra state" like it does for "state dict".
        #     Thus, we want to avoid putting extra state on the GPU
        #     since it may be loaded on the wrong device.
        # (3) The extra state consists of many small tensors. If we
        #     want to copy them all to CPU, then we need to avoid the
        #     overhead of many GPU-CPU memory transfers.
        """

        def to_cpu(src: torch.Tensor) -> torch.Tensor:
            """Helper function to make CPU copy of tensor

            Memory transfer is asynchronous w.r.t. host, so GPU should
            be synchronized before using result.

            """
            dst = torch.empty_like(src, device="cpu")
            dst.copy_(src, non_blocking=True)
            return dst

        # Store FP8 state if needed
        state = None
        fp8_checkpoint = self.fp8_meta["fp8_checkpoint"] or self.fp8 or self.fp8_calibration
        if not fp8_checkpoint:
            return torch.empty(0, dtype=torch.uint8)

        # Copy tensors to CPU and store
        state = {}
        recipe = self.fp8_meta["recipe"]
        if recipe.custom():
            # A CustomRecipe carries a user-supplied ``qfactory`` callable
            # (often a lambda or closure) that is not reliably picklable and
            # cannot be reconstructed on load. Persist only a lightweight
            # marker plus the numerical fields; the live recipe (with its
            # qfactory) is provided by the autocast context on resume and must
            # not be overwritten from the checkpoint. See ``set_extra_state``.
            state["recipe"] = {
                "__custom_recipe__": True,
                "fp8_format": recipe.fp8_format,
                "backward_override": getattr(recipe, "backward_override", None),
            }
        else:
            state["recipe"] = recipe

        if recipe.hif8_delayed():
            for mode in ("fwd", "bwd"):
                recipe_state = self.fp8_meta[f"scaling_{mode}"]
                state[f"hif8_state_{mode}"] = {
                    name: to_cpu(value) if value.device.type != "cpu" else value.clone()
                    for name, value in recipe_state.snapshot().items()
                }
        elif self._has_delayed_scaling_state_for_checkpoint():
            # Delayed scaling (native or composed inside a CustomRecipe) owns
            # running scale / amax_history buffers that must survive a
            # checkpoint. ``scaling_fwd`` / ``scaling_bwd`` expose ``.scale``
            # and ``.amax_history`` for both DelayedScalingRecipeState and
            # CustomRecipeState-with-DS.
            state["scale_fwd"] = to_cpu(self.fp8_meta["scaling_fwd"].scale)
            state["amax_history_fwd"] = to_cpu(self.fp8_meta["scaling_fwd"].amax_history)
            state["scale_bwd"] = to_cpu(self.fp8_meta["scaling_bwd"].scale)
            state["amax_history_bwd"] = to_cpu(self.fp8_meta["scaling_bwd"].amax_history)

        # Store other pickelable values
        extra = {}
        for k, v in self.fp8_meta.items():
            if k != "buffer_index_and_autocast_key" and isinstance(
                v, (bool, int, float, str, tuple, list)
            ):
                extra[k] = v
        if recipe.hif8_delayed():
            # Megatron's grouped-linear distributed checkpoint merge/split only
            # forwards ``recipe`` and ``extra_fp8_variables``. Reference the
            # snapshot there as well so it survives that path unchanged; pickle
            # memoization keeps a single copy in the serialized blob.
            for mode in ("fwd", "bwd"):
                extra[f"hif8_state_{mode}"] = state[f"hif8_state_{mode}"]
        state["extra_fp8_variables"] = extra

        # Serialize state into byte tensor
        torch.npu.synchronize()
        state_serialized = bytearray(pickle.dumps(state))
        state_serialized = torch.frombuffer(state_serialized, dtype=torch.uint8)
        return state_serialized

    def set_extra_state(self, state: torch.Tensor) -> None:
        """Load previous state."""
        # Maintain backwards compatibility with older checkpoints.
        if state is None:
            return

        # Load state
        if isinstance(state, torch.Tensor):
            # No FP8 is indicated by an empty tensor we don't need to unpickle.
            if state.numel() == 0:
                return
            # Default format: byte tensor with pickled data
            state = pickle.loads(state.detach().cpu().numpy().tobytes())  # nosec
        elif isinstance(state, io.BytesIO):
            # Deprecated format with io.BytesIO
            state.seek(0)
            state = torch.load(state, map_location="npu")
        else:
            raise RuntimeError("Unsupported checkpoint format.")

        if state is None:
            return

        # TE 1.x checkpoint compatibility: add DelayedScaling recipe if missing
        if "recipe" not in state:
            # TE 1.x only supported delayed scaling, which was the default recipe
            state["recipe"] = DelayedScaling()
            # TE 1.x also saved scale_inv, which is not needed with Recipe object
            state.pop("scale_inv_fwd", None)
            state.pop("scale_inv_bwd", None)

        # Load extra items. HiF8 snapshots may travel inside the opaque
        # ``extra_fp8_variables`` payload when Megatron merged a grouped linear
        # distributed checkpoint, since it only forwards that dict and recipe.
        extra_fp8_variables = dict(state["extra_fp8_variables"])
        hif8_states = {
            mode: extra_fp8_variables.pop(f"hif8_state_{mode}")
            for mode in ("fwd", "bwd")
            if f"hif8_state_{mode}" in extra_fp8_variables
        }
        self.fp8_meta.update(extra_fp8_variables)
        saved_recipe = state["recipe"]
        if isinstance(saved_recipe, dict) and saved_recipe.get("__custom_recipe__"):
            # CustomRecipe checkpoints only store a marker (the qfactory is not
            # picklable). Keep the live recipe supplied by the current autocast
            # context -- it already holds the qfactory needed to rebuild
            # quantizers -- and only restore the numerical buffers below.
            if not (self.fp8_meta.get("recipe") is not None and self.fp8_meta["recipe"].custom()):
                raise RuntimeError(
                    "Checkpoint was saved with a CustomRecipe but no CustomRecipe is "
                    "active on load. Re-create the module inside the same "
                    "autocast(recipe=CustomRecipe(...)) context before loading."
                )
        else:
            self.fp8_meta["recipe"] = saved_recipe
        if "global_fp8_buffer_pos_fwd_recompute" in self.fp8_meta:
            del self.fp8_meta["global_fp8_buffer_pos_fwd_recompute"]
        if self.fp8_meta["recipe"].hif8_delayed():
            self.fp8_meta.pop(FP8GlobalStateManager.get_hif8_buffer_info(), None)

        # Initialize before loading
        self.init_fp8_meta_tensors(self.fp8_meta["recipe"])

        def copy_tensor(src: torch.Tensor, dst: torch.Tensor) -> None:
            """Helper function to copy tensor from CPU

            Memory transfer is asynchronous w.r.t. host, so GPU should
            be synchronized before using result.

            """
            dst.copy_(src, non_blocking=True)

        # Load tensors
        if self.fp8_meta["recipe"].hif8_delayed():
            for mode in ("fwd", "bwd"):
                snapshot = state.get(f"hif8_state_{mode}")
                if snapshot is None:
                    snapshot = hif8_states.get(mode)
                if snapshot is None:
                    warnings.warn(
                        f"HiF8 checkpoint has no saved {mode} state; keeping the "
                        "freshly initialized policy state.",
                        stacklevel=2,
                    )
                    continue
                self.fp8_meta[f"scaling_{mode}"].load_snapshot(snapshot)
        elif self._has_delayed_scaling_state_for_checkpoint() and "scale_fwd" in state:
            copy_tensor(state["scale_fwd"], self.fp8_meta["scaling_fwd"].scale)
            copy_tensor(state["amax_history_fwd"], self.fp8_meta["scaling_fwd"].amax_history)
            copy_tensor(state["scale_bwd"], self.fp8_meta["scaling_bwd"].scale)
            copy_tensor(state["amax_history_bwd"], self.fp8_meta["scaling_bwd"].amax_history)
        torch.npu.synchronize()

    def set_activation_dtype(self, inp: torch.Tensor) -> None:
        """Get activation data type for AMP."""
        # Native AMP (`torch.autocast`) gets highest priority
        if torch.is_autocast_enabled():  # 这里暂时先定为float16, 暂时应不会跑进此分支
            self.fast_setattr("activation_dtype", torch.float16)
            return
        dtype = inp.dtype
        # All checks after this have already been performed once, thus skip
        if self.activation_dtype == dtype:
            return

        if not self.allow_different_data_and_param_types:
            for name, param in self.named_parameters():
                if param is not None:
                    if dtype != param.dtype:
                        raise TypeError(
                            "Data types for parameters must match when outside of autocasted "
                            f"region. Found input dtype: {dtype} and {name!r} dtype: {param.dtype}"
                        )
        self.fast_setattr("activation_dtype", dtype)

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = None
                  tensor parallel process group.
        """
        self.fast_setattr("tp_group", tp_group)
        self.fast_setattr("tp_group_initialized", True)

    def _get_fp8_params(self) -> Union[List[torch.Tensor], None]:
        """Returns the FP8 weights."""
        fp8_params = []
        for param in self.parameters(recurse=False):
            if isinstance(param, QuantizedTensor) and param.requires_grad:
                fp8_params.append(param)
        if len(fp8_params) == 0:
            return None
        return fp8_params

    def init_fp8_metadata(self, num_gemms: int = 1) -> None:
        """Initialize fp8 related metadata and tensors during fprop."""
        meta = self.fp8_meta

        fp8 = FP8GlobalStateManager.is_fp8_enabled()
        fp8_parameters = FP8GlobalStateManager.with_fp8_parameters()
        fp8_calibration = FP8GlobalStateManager.is_fp8_calibration()
        self.fast_setattr("fp8_parameters", fp8_parameters)
        self.fast_setattr("fp8", fp8)
        self.fast_setattr("fp8_calibration", fp8_calibration)
        fp8_enabled = fp8 or fp8_calibration
        meta["fp8_checkpoint"] = fp8_enabled

        _original_recipe = None

        if fp8_parameters or fp8_enabled:
            _original_recipe = meta.get("recipe", None)
            if self.fp8_initialized and FP8GlobalStateManager.get_fp8_recipe() == _original_recipe:
                # FP8 init has already been run and recipe is the same, don't do anything.
                return
            meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()
        else:
            # If fp8 isn't enabled, turn off and return.
            self.fast_setattr("fp8_initialized", False)
            return

        if fp8_parameters and not self.fp8_initialized:
            meta["num_gemms"] = num_gemms
            self.init_fp8_meta_tensors(meta["recipe"])

        if fp8_enabled:
            # Set FP8 and other FP8 metadata
            meta["num_gemms"] = num_gemms
            meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()

            # Set FP8_MAX per tensor according to recipe
            if hasattr(meta["recipe"], "fp8_format"):
                meta["fp8_max_fwd"] = meta["recipe"].fp8_format.value.fwd.value.max
                meta["fp8_max_bwd"] = meta["recipe"].fp8_format.value.bwd.value.max

            # Allocate scales and amaxes
            self.init_fp8_meta_tensors(meta["recipe"])
            self.fast_setattr("fp8_initialized", True)

            meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()

        _current_recipe = meta["recipe"]
        if _original_recipe is not None and not (
            issubclass(_current_recipe.__class__, _original_recipe.__class__)
            or issubclass(_original_recipe.__class__, _current_recipe.__class__)
        ):
            warnings.warn(
                f"Recipe type changed from {_original_recipe.__class__.__name__} "
                f"to {_current_recipe.__class__.__name__}. "
                "This may affect model behavior."
            )
            # Clear cached workspaces as they were created with the old recipe/quantizer type
            self._fp8_workspaces.clear()

    def prepare_forward(
        self,
        inp: torch.Tensor,
        num_gemms: int = 1,
        allow_non_contiguous: bool = False,
        allow_different_data_and_param_types: bool = False,
    ):
        self.fast_setattr(
            "allow_different_data_and_param_types", allow_different_data_and_param_types
        )
        self.fast_setattr("forwarded_at_least_once", True)
        # Activation recomputation is used and this is the second forward phase.
        if self.fp8 and in_fp8_activation_recompute_phase():
            FP8GlobalStateManager.get_old_fp8_meta_tensors_for_recompute(self.fp8_meta)
        else:
            if not (inp.is_cuda or getattr(inp, "is_npu", False)):
                raise RuntimeError(
                    f"TransformerEngine needs CUDA or NPU. Got input on device: {inp.device}"
                )

            if self.tp_size > 1:
                if not self.tp_group_initialized:
                    raise RuntimeError(
                        "Tensor parallel group not initialized. Call "
                        "set_tensor_parallel_group() before forward pass when tp_size > 1."
                    )

            self.set_activation_dtype(inp)
            self.init_fp8_metadata(num_gemms=num_gemms)
            self._check_weight_tensor_recipe_correspondence()

            hif8_delayed_recipe = self.fp8 and self.fp8_meta["recipe"].hif8_delayed()
            if hif8_delayed_recipe:
                if self.sequence_parallel and not self.fp8_meta["recipe"].reduce_amax:
                    raise ValueError(
                        "Amax reduction across tensor parallel group is "
                        "necessary when using sequence parallelism with FP8."
                    )
                if not FP8GlobalStateManager.fp8_graph_capturing():
                    FP8GlobalStateManager.register_hif8_states(self.fp8_meta)
                if self.training and is_fp8_activation_recompute_enabled():
                    FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(self.fp8_meta)

            # A CustomRecipe whose slots resolve to delayed scaling also needs
            # its amax tensors registered for reduction, so gate on the shared
            # helper (native DelayedScaling OR CustomRecipe-with-DS) rather than
            # recipe.delayed() alone.
            delayed_scaling_recipe = self.fp8 and FP8GlobalStateManager._has_delayed_scaling_state(
                self.fp8_meta
            )
            if delayed_scaling_recipe:
                if self.sequence_parallel:
                    # CustomRecipe has no top-level reduce_amax; read it from the
                    # inner delayed-scaling recipe when composed.
                    reduce_amax = getattr(self.fp8_meta["recipe"], "reduce_amax", None)
                    if reduce_amax is None:
                        inner = getattr(
                            self.fp8_meta.get("scaling_fwd"),
                            "_inner_delayed_scaling_recipe",
                            None,
                        )
                        reduce_amax = getattr(inner, "reduce_amax", True)
                    if not reduce_amax:
                        raise ValueError(
                            "Amax reduction across tensor parallel group is "
                            "necessary when using sequence parallelism with FP8."
                        )

                if not FP8GlobalStateManager.fp8_graph_capturing():
                    FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(self.fp8_meta)

                # Activation recomputation is used and this is the first forward phase.
                if self.training and is_fp8_activation_recompute_enabled():
                    FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(self.fp8_meta)

        if not allow_non_contiguous and not inp.is_contiguous():
            inp = inp.contiguous()
        return inp

    def end_forward(self):
        if not self.fp8:
            return
        if not (
            self.fp8_meta["recipe"].hif8_delayed()
            or FP8GlobalStateManager._has_delayed_scaling_state(self.fp8_meta)
        ):
            return
        if not in_fp8_activation_recompute_phase():
            return
        FP8GlobalStateManager.restore_fp8_meta_tensors(self.fp8_meta)

    @contextmanager
    def prepare_forward_ctx(
        self,
        inp: torch.Tensor,
        num_gemms: int = 1,
        allow_non_contiguous: bool = False,
        allow_different_data_and_param_types: bool = False,
    ) -> Generator[torch.Tensor, None, None]:
        """Checks and prepares for FWD execution."""
        inp = self.prepare_forward(
            inp, num_gemms, allow_non_contiguous, allow_different_data_and_param_types
        )
        try:
            yield inp
        finally:
            self.end_forward()

    def set_nccl_overlap_warning_if_tp(self) -> None:
        """When using TP, the NCCL communication needs to be scheduled
        before the GEMM for there to be a guaranteed overlap. From the
        host side in TE, the comm calls are always launched first, but
        to ensure that the GEMM isn't scheduled first, the environment
        variable `CUDA_DEVICE_MAX_CONNECTIONS` needs to be set to 1 to
        force a single channel.
        """
        if self.tp_size == 1:
            return
        num_cuda_work_queues = int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0"))
        if num_cuda_work_queues != 1:
            warnings.warn(
                "To guarantee overlapping TP and SP collectives with the backward"
                "GEMMs, set environment variable CUDA_DEVICE_MAX_CONNECTIONS = 1"
            )

    @staticmethod
    def grad_output_preprocess(
        ctx,
        grad_output: torch.Tensor,
        row_parallel_mode: bool,
        quantizer: Optional[Quantizer],
    ) -> Tuple[Union[torch.Tensor, QuantizedTensor], ...]:
        """Utility function for backward.
        Returns tuple in order (all optional/None based on training precion/recipe):
            R1: gathered `grad_output`.
            R2: bias gradient on R1.

        """
        grad_output = grad_output.reshape((-1, grad_output.shape[-1]))
        grad_output = grad_output.contiguous()

        if row_parallel_mode and ctx.sequence_parallel and not ctx.ub_overlap_ag:
            grad_output, _ = gather_along_dim(
                grad_output,
                ctx.tp_group,
            )

        grad_bias = None
        if ctx.use_bias:
            # Numerical-convention difference vs NVTE (REPORT atk 3#17): NVTE's
            # non-FP8 path computes db from the wgrad GEMM's cuBLASLt
            # CUBLASLT_EPILOGUE_BGRADB epilogue (gemm.cpp:241-245,
            # cublaslt_gemm.cu:669), which accumulates db in tiles inside the
            # GEMM and carries ~2% bf16 truncation noise on long columns. This
            # standalone sum is the exact fp32-accumulated column sum
            # (torch.sum on NPU accumulates in fp32 internally and rounds
            # once) - the numerically more accurate side. The difference is a
            # benign cross-implementation convention mismatch, not a defect;
            # the ATK linear guard therefore benchmarks grad_bias against an
            # exact fp64 column sum instead of the NVTE epilogue output
            # (tests/atk/module/linear/, REPORT atk 3#17 and 3#26). The wgrad
            # slot is the other output of that same fused GEMM and gets the
            # same treatment -- judged against exact fp64 it is clean here and
            # deviates on the NVTE side. The forward-output slot joined them
            # later for the same reason: on the two shapes that were still red
            # it is the NVTE side that deviates (0.9616/0.9851 vs exact fp64,
            # against 1.0000 here), not this one.
            grad_bias = grad_output.view((-1, grad_output.shape[-1])).sum(dim=0)

        use_fp8_bwd = ctx.fp8 and getattr(ctx, "backward_override", None) is None
        if not use_fp8_bwd and not ctx.debug:
            return grad_output, grad_bias

        # FP8 with all-gather: unfused bgrad, fused cast + transpose
        # Also supports debug quantization, which is handled inside gather_along_first_dim.
        # Quantize the gradient if needed
        if not isinstance(grad_output, QuantizedTensor) or (
            quantizer is not None and not quantizer.can_reuse(grad_output)
        ):
            grad_output = quantizer(grad_output)
        return grad_output, grad_bias

    def register_parameter(self, name, param, **kwargs):
        """
        Thin wrapper around PyTorch parameter registration to stash additional parameter
        metedata used in deferred initialization.
        """
        super().register_parameter(name, param)
        # Initialize param_init_meta exactly once during the init. FSDP2 can call
        # register parameter again to change parameters to DTensors. And it calls
        # it without custom fp8 specific kwargs that we need. And so we dont want
        # to reset/loose our fp8 init attributes.
        if hasattr(self, "param_init_meta") and name not in self.param_init_meta:
            self.param_init_meta[name] = _ParameterInitMeta(**kwargs)

    def reset_parameters(self, defer_init: Optional[bool] = False) -> None:
        """
        Reset all module parameters to initial values. Unless deferred initialization
        is specified, all parameters on a 'meta' device are also materialized on a real cuda
        device before the values are reset to initial.
        """
        if defer_init:
            return

        for name, param in self.named_parameters(recurse=False):
            # Check if parameter is a DTensor (FSDP2) or regular tensor
            is_dtensor = isinstance(param, DTensor)
            dtensor_param = param if is_dtensor else None
            # Need to update/quantize local tensor in case of DTensor
            param = param._local_tensor if is_dtensor else param
            # Ensure parameter is on a real device
            if param.device == torch.device("meta"):
                param = torch.empty_like(param, device="npu")
            # Initialize the parameter values on device
            init_fn = self.param_init_meta[name].init_fn
            get_rng_state_tracker = self.param_init_meta[name].get_rng_state_tracker
            if get_rng_state_tracker is None:
                init_fn(param)
            else:
                if hasattr(self, "rng_tracker_name") and self.rng_tracker_name:
                    with get_rng_state_tracker().fork(self.rng_tracker_name):
                        init_fn(param)
                else:
                    with get_rng_state_tracker().fork():
                        init_fn(param)

            # Wrap parameters in QuantizedTensor if needed
            fp8_meta_index = self.param_init_meta[name].fp8_meta_index
            high_precision_init_val = None
            if self.primary_weights_in_fp8 and fp8_meta_index is not None:
                # Keep high-precision values on CPU if needed
                if self.preserve_high_precision_init_val:
                    high_precision_init_val = param.detach().cpu()

                # Configure quantizer
                quantizer = self.quantizers["scaling_fwd"][fp8_meta_index]
                if quantizer is None:
                    raise RuntimeError("Weight quantizer has not been initialized")
                quantizer.set_usage(rowwise=True, columnwise=torch.is_grad_enabled())
                quantizer.internal = False
                # TODO @Muu 这里后续实现current的时候看下如何放开
                # if is_dtensor and isinstance(quantizer, Float8CurrentScalingQuantizer):
                #     device_mesh = dtensor_param.device_mesh
                #     amax_reduction_group = (
                #         device_mesh.get_group(mesh_dim="shard")
                #         if device_mesh.ndim > 1
                #         else device_mesh.get_group()
                #     )
                #     quantizer.amax_reduction_group = amax_reduction_group
                #     quantizer.with_amax_reduction = True
                # Quantize parameter
                param = quantizer(param)

            # Redo parameter wrap in case we broke it above
            # NOTE: Currently this can only be broken when primary weights are in Fp8 but
            #       re-applying the nn.Parameter() wrap is a no-op when the input is already
            #       a parameter so we always re-apply it just for extra safety.
            if is_dtensor:
                # recreate the DTensor from the parameter.
                dtensor_param = DTensor.from_local(
                    param,
                    device_mesh=dtensor_param.device_mesh,
                    placements=dtensor_param.placements,
                    shape=dtensor_param.size(),
                    stride=dtensor_param.stride(),
                )
                dtensor_param = torch.nn.Parameter(dtensor_param)
            else:
                param = torch.nn.Parameter(param)

            # Keep high-precision values on CPU if needed
            if high_precision_init_val is not None:
                # - Master weights are initialized from model weights, if we use fp8 primary
                #   weights to initialize master weights, the numerical values of master weights
                #   are not consistent with the numerical values when we initialize them from
                #   bf16/fp16 weights.
                # - So we add a `_high_precision_init_val` attribute to each model weight to store
                #   the original bf16/fp16 weight on cpu before casting it to fp8. And users can
                #   use `get_high_precision_init_val` to get this cpu tensor.
                # - This cpu tensor is not needed once the master weight is initialized, so users
                #   should call `clear_high_precision_init_val` to remove it after master weight
                #   is initialized.

                def get(self):
                    if hasattr(self, "_high_precision_init_val"):
                        return self._high_precision_init_val
                    return None

                def clear(self):
                    if hasattr(self, "_high_precision_init_val"):
                        del self._high_precision_init_val

                param._high_precision_init_val = high_precision_init_val
                param.get_high_precision_init_val = MethodType(get, param)
                param.clear_high_precision_init_val = MethodType(clear, param)
                # Update the parameter based on its type

            if not is_dtensor:
                self.module_setattr(name, param)
            else:
                self.module_setattr(name, dtensor_param)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        This function loads tensors and extra state including fp8 metadata.
        This metadata is essential for copying fp8 tensors, as the copy_ function
        uses the scale_inv parameter from fp8_meta to set the correct scaling factor
        for the new tensor.
        Hence, this extra state must be loaded before the tensor copying process,
        not after, as is typically done in _load_from_state_dict.
        Tensors are copied into fp8 tensors only when self.primary_weights_in_fp8=True,
        otherwise, this behavior is not required.
        """
        if self.primary_weights_in_fp8:
            extra_state_key = prefix + torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX
            if extra_state_key in state_dict:
                self.set_extra_state(state_dict[extra_state_key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def register_wgrad_accumulation_and_reduce_hooks(self, wgrad_accumulation_and_reduce_hook):
        """
        This method is used to manually control the weight gradient accumulation and reduce.
        This method should be called before the backward() method.
        Set the skip_wgrad_accumulation_and_reduce to True to skip the weight gradient accumulation
        and reduce in backward();
        And register the wgrad_accumulation_and_reduce_func to be called in backward_dw() method.
        """
        self.wgrad_accumulation_and_reduce_hooks.append(wgrad_accumulation_and_reduce_hook)

    def need_backward_dw(self):
        """
        Check if this module needs to execute the delayed weight gradient computation.
        This method should be used at the beginning of self.backward_dw() to determine if it
        should actually be executed or just return without doing anything.
        User can also manually call this method to check that before calling into backward_dw().
        """
        return self.wgrad_store is not None and self.wgrad_store.delay_wgrad_compute()

    def backward_dw(self):
        """
        Execute the delayed weight gradient computation.
        This method is called after the main backward pass to compute weight gradients.
        """
        if not self.need_backward_dw():
            return
        (wgrad, bgrad), _ = self.wgrad_store.pop()
        if not self.fuse_wgrad_accumulation:
            weight_tensor = noop_cat(self._get_weight_tensors())
            weight_tensor.grad = wgrad.to(weight_tensor.dtype)
        if self.use_bias and bgrad is not None:
            # TODO @Muu 这里bgrad 看下后续怎么解开
            bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
            if bias_tensor.grad is None:
                bias_tensor.grad = bgrad.to(bias_tensor.dtype)
        del wgrad
        del bgrad
        self._trigger_wgrad_accumulation_and_reduce_hooks()

    def _trigger_wgrad_accumulation_and_reduce_hooks(self):
        """
        Trigger the wgrad accumulation and reduce hooks.
        """
        for wgrad_accumulation_and_reduce_hook in self.wgrad_accumulation_and_reduce_hooks:
            wgrad_accumulation_and_reduce_hook()
