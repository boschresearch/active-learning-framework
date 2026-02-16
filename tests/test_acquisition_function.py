import numpy as np
import pytest
from alef.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from alef.configs.acquisition_function import (
    BasicRandomConfig,
    BasicSafeRandomConfig,
    BasicPredVarianceConfig,
    BasicPredSigmaConfig,
    BasicPredEntropyConfig,
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
    BasicSafeDiscoverConfig,
    BasicSafeDiscoverQuantileConfig,
    BasicSafeOptConfig,
    BasicSafeGPUCBConfig,
    BasicEIConfig,
    BasicSafeEIConfig,
    BasicSafeDiscoverOptConfig,
    BasicSafeDiscoverOptQuantileConfig,
)
from alef.acquisition_function import (
    Random,
    SafeRandom,
    PredVariance,
    PredSigma,
    PredEntropy,
    SafePredEntropy,
    SafePredEntropyAll,
    SafeDiscover,
    SafeDiscoverQuantile,
    EI,
    SafeEI,
    SafeOpt,
    SafeGPUCB,
    SafeDiscoverOpt,
    SafeDiscoverOptQuantile,
)


from alef.configs.config_picker import ConfigPicker
from alef.models.model_factory import ModelFactory

kernel_config = ConfigPicker.pick_kernel_config("Matern52WithPriorConfig")(
    input_dimension=3, fix_variance=True, add_prior=False
)
model_config = ConfigPicker.pick_model_config("BasicGPModelConfig")(
    kernel_config=kernel_config, observation_noise=0.1, optimize_hps=False
)
model = ModelFactory.build(model_config)
N = 50
N_test = 20
X = np.random.normal(0, 0.5, size=[N + N_test, 3])
Y = np.exp(X[:, [0]] ** 2 - np.sin(X[:, [1]]) + X[:, [2]]) + np.random.normal(0, 0.2, size=[N + N_test, 1])


Xtrain = X[:N]
Ytrain = Y[:N]
Xt = X[N:]
Yt = Y[N:]
model.infer(Xtrain, Ytrain)


config2function = {
    0: (BasicRandomConfig, Random),
    1: (BasicPredVarianceConfig, PredVariance),
    2: (BasicPredSigmaConfig, PredSigma),
    3: (BasicPredEntropyConfig, PredEntropy),
    4: (BasicEIConfig, EI),
}

safe_config2function = {
    0: (BasicSafeRandomConfig, SafeRandom),
    1: (BasicSafePredEntropyConfig, SafePredEntropy),
    2: (BasicSafePredEntropyAllConfig, SafePredEntropyAll),
    3: (BasicSafeDiscoverConfig, SafeDiscover),
    4: (BasicSafeDiscoverQuantileConfig, SafeDiscoverQuantile),
    5: (BasicSafeDiscoverOptConfig, SafeDiscoverOpt),
    6: (BasicSafeDiscoverOptQuantileConfig, SafeDiscoverOptQuantile),
    7: (BasicSafeOptConfig, SafeOpt),
    8: (BasicSafeGPUCBConfig, SafeGPUCB),
    9: (BasicSafeEIConfig, SafeEI),
}


@pytest.mark.parametrize(
    "config,acquisition_class",
    [(config_class(), function_class) for config_class, function_class in config2function.values()]
    + [
        (config_class(safety_thresholds_lower=-np.inf, safety_thresholds_upper=np.inf), function_class)
        for config_class, function_class in safe_config2function.values()
    ],
)
def test_acquisition_function_factory(config, acquisition_class):
    assert isinstance(AcquisitionFunctionFactory.build(config), acquisition_class)


@pytest.mark.parametrize("config", [config_class() for config_class, _ in config2function.values()])
def test_acquisition_score(config):
    acq_object = AcquisitionFunctionFactory.build(config)
    score = acq_object.acquisition_score(
        Xt,
        model,
        safety_models=None,
        x_data=Xtrain,
        y_data=Ytrain,
    )
    assert score.shape == (N_test,)


@pytest.mark.parametrize(
    "config",
    [
        config_class(safety_thresholds_lower=-np.inf, safety_thresholds_upper=np.inf)
        for config_class, _ in safe_config2function.values()
    ],
)
def test_acquisition_safe_set(config):
    acq_object = AcquisitionFunctionFactory.build(config)
    S = acq_object.compute_safe_set(Xt, safety_models=[model])
    assert S.shape == (N_test,)
    assert S.dtype == bool


@pytest.mark.parametrize(
    "config",
    [
        config_class(safety_thresholds_lower=-np.inf, safety_thresholds_upper=np.inf)
        for config_class, _ in safe_config2function.values()
    ],
)
def test_acquisition_safe_score(config):
    acq_object = AcquisitionFunctionFactory.build(config)
    score, S = acq_object.acquisition_score(
        Xt, model, safety_models=None, x_data=Xtrain, y_data=Ytrain, return_safe_set=True
    )
    assert score.shape == (N_test,)
    assert S.shape == (N_test,)
