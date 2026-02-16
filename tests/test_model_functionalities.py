import time
import torch
from alef.configs.kernels.linear_configs import BasicLinearConfig, LinearWithPriorConfig
from alef.configs.kernels.matern32_configs import Matern32WithPriorConfig
from alef.configs.kernels.periodic_configs import BasicPeriodicConfig, PeriodicWithPriorConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicLinearKernelPytorchConfig,
    BasicPeriodicKernelPytorchConfig,
    BasicRBFPytorchConfig,
    BasicRQKernelPytorchConfig,
    LinearWithPriorPytorchConfig,
    Matern32WithPriorPytorchConfig,
    Matern52WithPriorPytorchConfig,
    PeriodicWithPriorPytorchConfig,
    RBFWithPriorPytorchConfig,
    RQWithPriorPytorchConfig,
)
from alef.configs.kernels.rational_quadratic_configs import BasicRQConfig, RQWithPriorConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.configs.kernels.spectral_mixture_kernel_config import BasicSMKernelConfig
from alef.configs.models.gp_model_kernel_search_config import GPBigNDimFullKernelSearchConfig
from alef.configs.models.gp_model_pytorch_config import BasicGPModelPytorchConfig, GPModelPytorchMAPConfig
from alef.configs.models.gp_model_scalable_config import BasicScalableGPModelConfig, GPRAdamWithValidationSet
from alef.models.model_factory import ModelFactory
from alef.configs.kernels.hhk_configs import (
    HHKFourLocalDefaultConfig,
    HHKTwoLocalDefaultConfig,
)
from alef.configs.models.gp_model_config import BasicGPModelConfig, GPModelFastConfig, GPModelWithNoisePriorConfig
from alef.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from alef.configs.models.gp_model_marginalized_config import (
    BasicGPModelMarginalizedConfig,
    GPModelMarginalizedConfigMAPInitialized,
)
from alef.configs.kernels.matern52_configs import Matern52WithPriorConfig
from alef.oracles.safe_test_func import SafeTestFunc
from alef.enums.global_model_enums import PredictionQuantity as PredictionQuantMarg
from alef.enums.global_model_enums import InitialParameters
import pytest
import numpy as np

from alef.utils.utils import print_gpytorch_parameters


@pytest.mark.parametrize(
    "config_class", (BasicGPModelConfig, BasicGPModelLaplaceConfig, BasicGPModelMarginalizedConfig)
)
def test_gp_models_inference(config_class):
    oracle = SafeTestFunc(0.01)
    x_data, y_data = oracle.get_random_data(20)
    x_test, _ = oracle.get_random_data(10)
    kernel_config = Matern52WithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = config_class(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    gp_model.infer(x_data, y_data)
    gp_model.predictive_dist(x_test)


def test_gp_model_number_of_parameters():
    oracle = SafeTestFunc(0.01)
    x_data, y_data = oracle.get_random_data(20)
    kernel_config = HHKTwoLocalDefaultConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    gp_model.build_model(x_data, y_data)
    num_parameters = gp_model.get_number_of_trainable_parameters()
    assert num_parameters == 8
    kernel_config = HHKFourLocalDefaultConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    gp_model.build_model(x_data, y_data)
    num_parameters = gp_model.get_number_of_trainable_parameters()
    assert num_parameters == 18


def test_scalable_gp_models():
    oracle = SafeTestFunc(0.01)
    x_data, y_data = oracle.get_random_data(40)
    x_test, _ = oracle.get_random_data(10)
    kernel_config = Matern52WithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = BasicScalableGPModelConfig(kernel_config=kernel_config, n_iterations=300, n_repeats=2)
    gp_model = ModelFactory.build(model_config)
    gp_model.infer(x_data, y_data)
    gp_model.predictive_dist(x_test)
    model_config = GPRAdamWithValidationSet(kernel_config=kernel_config, n_iterations=300, n_repeats=2)
    gp_model = ModelFactory.build(model_config)
    gp_model.infer(x_data, y_data)
    gp_model.predictive_dist(x_test)


@pytest.mark.parametrize(
    "start_parameters_class,kernel_config_class",
    [
        (InitialParameters.PERTURB, Matern52WithPriorConfig),
        (InitialParameters.UNIFORM_DISTRIBUTION, BasicRBFConfig),
        (InitialParameters.UNIFORM_DISTRIBUTION, HHKTwoLocalDefaultConfig),
    ],
)
def test_gp_model_mutlistart(start_parameters_class, kernel_config_class):
    oracle = SafeTestFunc(0.01)
    x_data, y_data = oracle.get_random_data(20)
    x_test, _ = oracle.get_random_data(10)
    kernel_config = kernel_config_class(input_dimension=oracle.get_dimension())
    model_config = BasicGPModelConfig(kernel_config=kernel_config, initial_parameter_strategy=start_parameters_class)
    gp_model = ModelFactory.build(model_config)
    time_before = time.perf_counter()
    gp_model.infer(x_data, y_data)
    time_after = time.perf_counter()
    print(time_after - time_before)
    infer_time = gp_model.get_last_inference_time()
    print(infer_time)
    assert np.isclose(
        gp_model.training_loss(), np.min(gp_model.multi_start_losses), rtol=1e-05, atol=1e-08, equal_nan=False
    )


def test_gp_marginalized_model_special_funcs():
    oracle = SafeTestFunc(0.01)
    x_data, y_data = oracle.get_random_data(20)
    x_test, y_test = oracle.get_random_data(20)
    kernel_config = Matern52WithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelMarginalizedConfigMAPInitialized(kernel_config=kernel_config)
    model_config.prediction_quantity = PredictionQuantMarg.PREDICT_Y
    gp_model = ModelFactory.build(model_config)
    gp_model.infer(x_data, y_data)
    pred_mu, pred_sigma = gp_model.predictive_dist(x_test)
    gp_model.entropy_predictive_dist(x_test)
    parameter_samples, _ = gp_model.get_posterior_samples()
    pred_dists = gp_model.get_predictive_distributions(x_test)
    counter = 0
    pred_mu_list = []
    for model in gp_model.yield_posterior_models():
        pred_mu1, pred_sigma1 = model.predict_y(x_test)
        pred_mu2, pred_sigma2 = pred_dists[counter]
        pred_mu_list.append(pred_mu2)
        assert np.allclose(np.squeeze(pred_mu1), np.squeeze(pred_mu2), rtol=1e-05, atol=1e-08)
        assert np.allclose(np.squeeze(np.sqrt(pred_sigma1)), np.squeeze(pred_sigma2), rtol=1e-05, atol=1e-08)
        for i, parameters_yielded in enumerate(model.trainable_parameters):
            parameters = parameter_samples[i][counter]

            assert np.allclose(parameters_yielded.numpy(), parameters.numpy(), rtol=1e-05, atol=1e-08)
        counter += 1
    pred_mu_avg = np.mean(np.array(pred_mu_list), axis=0)
    assert np.allclose(pred_mu_avg, pred_mu)


def test_gp_model_reset():
    oracle = SafeTestFunc(0.01)
    x_data, y_data = oracle.get_random_data(20)
    kernel_config = Matern52WithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    lengthscale_before_training = gp_model.kernel.kernel.lengthscales.numpy()
    variance_before_training = gp_model.kernel.kernel.variance.numpy()
    time_before = time.perf_counter()
    gp_model.infer(x_data, y_data)
    time_after = time.perf_counter()
    print(time_after - time_before)
    print(gp_model.get_last_inference_time())
    lengthscale_after_training = gp_model.kernel.kernel.lengthscales.numpy()
    variance_after_training = gp_model.kernel.kernel.variance.numpy()
    assert not np.isclose(lengthscale_before_training, lengthscale_after_training)
    assert not np.isclose(variance_before_training, variance_after_training)
    gp_model.reset_model()
    lengthscale_after_reset = gp_model.kernel.kernel.lengthscales.numpy()
    variance_after_reset = gp_model.kernel.kernel.variance.numpy()
    assert np.isclose(lengthscale_before_training, lengthscale_after_reset)
    assert np.isclose(variance_before_training, variance_after_reset)


def test_gp_model_with_input_initialized_kernel():
    oracle = SafeTestFunc(0.01)
    x_data, y_data = oracle.get_random_data(20)
    kernel_config = BasicSMKernelConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    gp_model.infer(x_data, y_data)
    assert len(gp_model.kernel_initial_parameter_cache.parameters_list) == 1
    gp_model.infer(x_data, y_data)
    assert len(gp_model.kernel_initial_parameter_cache.parameters_list) == 1
    gp_model.reset_model()
    gp_model.infer(x_data, y_data)
    assert len(gp_model.kernel_initial_parameter_cache.parameters_list) == 1
    model_config = BasicGPModelConfig(kernel_config=kernel_config)
    gp_model = ModelFactory.build(model_config)
    gp_model.infer(x_data, y_data)


def test_gpytorch_model_fixed_noise_and_fixed_hps():
    pass


@pytest.mark.parametrize(
    "pytorch_kernel_config,gpflow_kernel_config",
    [
        (BasicRBFPytorchConfig, BasicRBFConfig),
        (RBFWithPriorPytorchConfig, RBFWithPriorConfig),
        (BasicPeriodicKernelPytorchConfig, BasicPeriodicConfig),
        (PeriodicWithPriorPytorchConfig, PeriodicWithPriorConfig),
        (BasicRQKernelPytorchConfig, BasicRQConfig),
        (RQWithPriorPytorchConfig, RQWithPriorConfig),
        (BasicLinearKernelPytorchConfig, BasicLinearConfig),
        (LinearWithPriorPytorchConfig, LinearWithPriorConfig),
        (Matern52WithPriorPytorchConfig, Matern52WithPriorConfig),
        (Matern32WithPriorPytorchConfig, Matern32WithPriorConfig),
    ],
)
def test_marginal_likelihood_pytorch_gp_model(pytorch_kernel_config, gpflow_kernel_config):
    torch.set_default_dtype(torch.float64)
    oracle = SafeTestFunc(0.01)
    n_data = 20
    x_data, y_data = oracle.get_random_data(n_data)
    model_config = BasicGPModelPytorchConfig(kernel_config=pytorch_kernel_config(input_dimension=1))
    model_config_gpflow = BasicGPModelConfig(kernel_config=gpflow_kernel_config(input_dimension=1))
    model_config.initial_likelihood_noise = 0.01
    model = ModelFactory.build(model_config)
    model_gpflow = ModelFactory.build(model_config_gpflow)
    model.build_model(torch.from_numpy(x_data), torch.from_numpy(np.squeeze(y_data)))
    print_gpytorch_parameters(model.model)
    log_posterior_density = model.eval_log_posterior_density(x_data, np.squeeze(y_data)).detach().numpy()
    model_gpflow.build_model(x_data, y_data)
    log_posterior_density_gpflow = model_gpflow.model.log_posterior_density().numpy()
    log_posterior_density_gpytorch = log_posterior_density * n_data
    assert np.allclose(log_posterior_density_gpytorch, log_posterior_density_gpflow)
    mll = model.eval_log_marginal_likelihood(x_data, np.squeeze(y_data)).detach().numpy()
    mll_gpflow = model_gpflow.model.log_marginal_likelihood().numpy()
    assert np.allclose(mll * n_data, mll_gpflow)
    model_config_with_noise_prior = GPModelPytorchMAPConfig(kernel_config=pytorch_kernel_config(input_dimension=1))
    model_config_gpflow_with_noise_prior = GPModelWithNoisePriorConfig(
        kernel_config=gpflow_kernel_config(input_dimension=1)
    )
    model_config_with_noise_prior.initial_likelihood_noise = 0.01

    model_with_noise_prior = ModelFactory.build(model_config_with_noise_prior)
    model_gpflow_with_noise_prior = ModelFactory.build(model_config_gpflow_with_noise_prior)
    model_with_noise_prior.build_model(torch.from_numpy(x_data), torch.from_numpy(np.squeeze(y_data)))
    print_gpytorch_parameters(model_with_noise_prior.model)
    log_posterior_density_with_noise_prior = (
        model_with_noise_prior.eval_log_posterior_density(x_data, np.squeeze(y_data)).detach().numpy()
    )
    model_gpflow_with_noise_prior.build_model(x_data, y_data)
    log_posterior_density_gpflow_with_noise_prior = model_gpflow_with_noise_prior.model.log_posterior_density().numpy()
    assert np.allclose(log_posterior_density_with_noise_prior * n_data, log_posterior_density_gpflow_with_noise_prior)
    mll_with_noise_prior = (
        model_with_noise_prior.eval_log_marginal_likelihood(x_data, np.squeeze(y_data)).detach().numpy()
    )
    mll_gpflow_with_noise_prior = model_gpflow_with_noise_prior.model.log_marginal_likelihood().numpy()
    assert np.allclose(mll_with_noise_prior * n_data, mll_gpflow_with_noise_prior)

    assert np.allclose(mll_with_noise_prior, mll)
    assert np.allclose(mll_gpflow_with_noise_prior, mll_gpflow)
    assert not np.allclose(log_posterior_density, log_posterior_density_with_noise_prior)
    assert not np.allclose(log_posterior_density_gpflow, log_posterior_density_gpflow_with_noise_prior)
    torch.set_default_dtype(torch.float32)


def test_gp_kernel_search():
    oracle = SafeTestFunc(0.01)
    n_data = 10
    x_data, y_data = oracle.get_random_data(n_data)
    model_config = GPBigNDimFullKernelSearchConfig(input_dimension=1)
    model_config.n_steps_bo: int = 2
    model_config.fast_inference = True
    model = ModelFactory.build(model_config)
    model.infer(x_data, y_data)


if __name__ == "__main__":
    test_gp_kernel_search()
