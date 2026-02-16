from alef.configs.config_picker import ConfigPicker
from alef.configs.models.gp_model_marginalized_config import (
    BasicGPModelMarginalizedConfig,
    GPModelMarginalizedConfigMoreSamplesConfig,
    GPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
    GPModelMarginalizedConfigMoreThinningConfig,
    GPModelMarginalizedConfigMAPInitialized,
)
from alef.configs.models.gp_model_config import BasicGPModelConfig, GPModelFastConfig, GPModelFixedNoiseConfig
from alef.configs.kernels.hhk_configs import HHKEightLocalDefaultConfig
from alef.configs.active_learner.batch_active_learner_configs import EntropyBatchActiveLearnerConfig


def test_picker():
    picker = ConfigPicker()
    config = picker.pick_model_config("BasicGPModelMarginalizedConfig")
    assert config == BasicGPModelMarginalizedConfig
    config = picker.pick_model_config("GPModelMarginalizedConfigMoreSamplesConfig")
    assert config == GPModelMarginalizedConfigMoreSamplesConfig
    config = picker.pick_model_config("GPModelMarginalizedConfigMoreSamplesMoreThinningConfig")
    assert config == GPModelMarginalizedConfigMoreSamplesMoreThinningConfig
    config = picker.pick_model_config("GPModelMarginalizedConfigMoreThinningConfig")
    assert config == GPModelMarginalizedConfigMoreThinningConfig
    config = picker.pick_model_config("BasicGPModelConfig")
    assert config == BasicGPModelConfig
    config = picker.pick_kernel_config("HHKEightLocalDefaultConfig")
    assert config == HHKEightLocalDefaultConfig
    assert not config == BasicGPModelConfig
    config = picker.pick_model_config("GPModelMarginalizedConfigMAPInitialized")
    assert config == GPModelMarginalizedConfigMAPInitialized
    config = picker.pick_model_config("GPModelFastConfig")
    assert config == GPModelFastConfig
    config = picker.pick_batch_active_learner_config("EntropyBatchActiveLearnerConfig")
    assert config == EntropyBatchActiveLearnerConfig
    config = picker.pick_model_config("GPModelFixedNoiseConfig")
    assert config == GPModelFixedNoiseConfig
