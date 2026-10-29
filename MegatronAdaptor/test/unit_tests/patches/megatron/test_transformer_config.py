# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""transformer_config_init_wrapper 的 None 注入回归测试。

wrapper 的全量 CLI 注入不得用 CLI 缺省的 None 覆盖 dataclass 的非 None 默认值（如 MLATransformerConfig.rope_type 默认 "yarn"），
同时必须保留上游在 kw_args 中显式携带的 None（覆盖语义，如非 GQA 时的num_query_groups）与未知 CLI 字段的动态挂载能力。

用一个与 TransformerConfig 同构的 dummy dataclass 直接驱动 wrapper，不依赖 NPU / 完整训练链路。
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import pytest

from megatron_adaptor.patches.megatron import transformer_config as tc_patch


@dataclass
class _DummyConfig:
    # 模拟 MLATransformerConfig.rope_type：dataclass 默认非 None，CLI 默认 None
    rope_type: str = "yarn"
    # 模拟 TransformerConfig.num_query_groups：上游会显式注入 None 覆盖 CLI 值
    num_query_groups: Optional[int] = None
    hidden_size: int = 256


def _build_config(monkeypatch, full_args: argparse.Namespace, **kwargs):
    """经 wrapper 构造 _DummyConfig，模拟 config_class(**kw_args) 现场。"""
    monkeypatch.setattr(tc_patch, "get_full_args", lambda: full_args)
    wrapped_init = tc_patch.transformer_config_init_wrapper(_DummyConfig.__init__)
    obj = _DummyConfig.__new__(_DummyConfig)
    wrapped_init(obj, **kwargs)
    return obj


class TestTransformerConfigInitWrapper:
    def test_missing_cli_value_keeps_dataclass_default(self, monkeypatch):
        """CLI 缺省 --rope-type（None），上游已 pop，wrapper 不得重新注入 None 覆盖默认值 "yarn"。"""
        full_args = argparse.Namespace(rope_type=None, hidden_size=512)
        # kwargs 对应上游 core_transformer_config_from_args 装配结果：
        # rope_type 已被 pop，hidden_size 保留
        cfg = _build_config(monkeypatch, full_args, hidden_size=512)
        assert cfg.rope_type == "yarn"
        # 非 None 的 CLI 值仍应正常注入
        assert cfg.hidden_size == 512

    def test_explicit_none_in_kwargs_is_preserved(self, monkeypatch):
        """kwargs 中显式的 None 是上游覆盖语义（非 GQA 时 kw_args['num_query_groups'] = None，而 CLI 自动默认为 1），
        必须透传，不能被跳过、也不能被 full_args 的值顶掉。
        """
        full_args = argparse.Namespace(num_query_groups=1, rope_type=None)
        cfg = _build_config(monkeypatch, full_args, num_query_groups=None)
        assert cfg.num_query_groups is None

    def test_explicit_cli_value_passthrough(self, monkeypatch):
        """CLI 显式传参时值原样生效（经 kwargs 或 full_args 均可）。"""
        full_args = argparse.Namespace(rope_type="rope")
        cfg = _build_config(monkeypatch, full_args, rope_type="rope")
        assert cfg.rope_type == "rope"

    def test_cli_value_fills_gap_when_kwargs_absent(self, monkeypatch):
        """kwargs 未覆盖的已知字段仍由 full_args 补齐（直接构造 config、不走 core_transformer_config_from_args 的路径依赖该行为）。"""
        full_args = argparse.Namespace(rope_type=None, hidden_size=1024)
        cfg = _build_config(monkeypatch, full_args)
        assert cfg.hidden_size == 1024
        assert cfg.rope_type == "yarn"

    def test_unknown_cli_fields_still_attached(self, monkeypatch):
        """MA 自有/动态 CLI 字段仍被挂载到 config 实例上；None 值的未知字段保持既有行为（挂载为 None），避免依赖 getattr 的特性校验代码回归。"""
        full_args = argparse.Namespace(
            rope_type=None,
            ma_extra_feature="on",
            ma_optional=None,
        )
        cfg = _build_config(monkeypatch, full_args)
        assert getattr(cfg, "ma_extra_feature") == "on"
        assert hasattr(cfg, "ma_optional")
        assert getattr(cfg, "ma_optional") is None

    @pytest.mark.parametrize("owned_key", sorted(tc_patch._MODEL_CONFIG_OWNED_KEYS))
    def test_model_config_owned_fields_are_not_injected(self, monkeypatch, owned_key):
        """模型级配置（GPTModelConfig/HybridModelConfig）专属字段不得注入 TransformerConfig。

        一旦注入，模型级配置的 __setattr__ 会把赋值转发给 transformer，而读取会回落到
        dataclass 类默认值（如 position_embedding_type 读出 learned_absolute、vocab_size
        读出 None），导致模型结构或构建参数错误。参数化覆盖整个排除集合，新增条目自动纳入。
        """
        full_args = argparse.Namespace(
            rope_type=None,
            ma_extra_feature="on",  # 普通未知 CLI 字段：仍应挂载
            **{owned_key: "x"},
        )
        cfg = _build_config(monkeypatch, full_args)
        assert not hasattr(cfg, owned_key)
        assert getattr(cfg, "ma_extra_feature") == "on"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
