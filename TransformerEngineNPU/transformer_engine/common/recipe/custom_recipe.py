# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import dataclass
from typing import Any, Callable, Optional

from transformer_engine.common.recipe import Format
from transformer_engine.common.recipe.base import (
    Recipe,
    _default_backward_override,
    _validate_backward_override,
)


@dataclass(repr=False)
class CustomRecipe(Recipe):
    """Custom recipe that allows users to provide quantizer factories.

    .. warning::
        **EXPERIMENTAL**: Custom recipe is experimental, still under active development,
        and the API is subject to change without notice. Use at your own risk.

    Parameters
    ----------
    qfactory : Callable
        Factory callable that returns a quantizer instance for a given
        ``QuantizerRole``. The callable is invoked as::

            qfactory(role: QuantizerRole) -> Quantizer

        ``QuantizerRole`` is a frozen dataclass with the following fields:

        - ``module_type`` (str): module type, e.g. ``"linear"``, ``"grouped_linear"``
        - ``tensor_type`` (str): tensor role, e.g. ``"input"``, ``"weight"``, ``"grad_output"``
        - ``name`` (str): module instance name, e.g. ``"qkv"``, ``"fc1"``

    fp8_format : Format, default = Format.HYBRID
        FP8 format to use for ``get_fp8_te_dtype`` and related helpers.
    backward_override : {None, 'high_precision', 'dequantized'}, default = None
        Backward precision mode. None does not modify backward behavior,
        ``high_precision`` keeps original high-precision operands for backward,
        and ``dequantized`` dequantizes saved operands to the active high-precision
        compute dtype (e.g. BF16/FP16/FP32) for backward.
    """

    qfactory: Callable[..., Any]
    fp8_format: Format = Format.HYBRID
    fp8_dpa: bool = False
    fp8_mha: bool = False
    backward_override: Optional[str] = _default_backward_override()

    def __post_init__(self) -> None:
        _validate_backward_override(self.backward_override)

    @classmethod
    def custom(cls):
        return True

    def _make_repr(self) -> str:
        return (
            f"recipe_type={self.__class__.__name__}, "
            f"qfactory={self.qfactory}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"backward_override={self.backward_override}"
        )
