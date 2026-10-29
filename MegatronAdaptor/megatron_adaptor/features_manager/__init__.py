from megatron_adaptor.features_manager.features_manager import FeaturesManager
from megatron_adaptor.features_manager.megatron.megatron_basic import MegatronBasicFeature
from megatron_adaptor.features_manager.megatron_fix.megatron_fix import MegatronFixFeature
from megatron_adaptor.features_manager.requirements.requirements import RequirementsBasicFeature
from megatron_adaptor.features_manager.transformer_engine.transformer_engine import TransformerEngineBasicFeature
from megatron_adaptor.features_manager.functional.npu_deterministic import NPUDeterministicFeature
from megatron_adaptor.features_manager.functional.profiler_default import ProfilerDefaultFeature
from megatron_adaptor.features_manager.functional.profile import ProfileFeature
from megatron_adaptor.features_manager.functional.npu_datadump import NPUDataDumpFeature
from megatron_adaptor.auto_execute import AutoExecuteFunction


def create_features_list() -> list:
    """Create the built-in MegatronAdaptor feature list."""
    return [
        RequirementsBasicFeature(),
        MegatronFixFeature(),
        MegatronBasicFeature(),
        TransformerEngineBasicFeature(),
        NPUDeterministicFeature(),
        ProfilerDefaultFeature(),
        ProfileFeature(),
        NPUDataDumpFeature(),
    ]


FEATURE_LIST = create_features_list()


@AutoExecuteFunction
def set_default_features_list():
    """Install MA's built-in feature list unless an inheritor suppresses it."""
    FeaturesManager.set_features_list(FEATURE_LIST)


set_default_features_list()

__all__ = ["FEATURE_LIST", "create_features_list", "set_default_features_list"]
