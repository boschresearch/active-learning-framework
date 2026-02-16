from alef.configs.kernels.grammar_tree_kernel_kernel_configs import OTWeightedDimsExtendedGrammarKernelConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig
from alef.configs.models.deep_gp_config import DeepGPConfig
from alef.configs.models.object_gp_model_config import BasicObjectGPModelConfig
from alef.models.deep_gp import DeepGP
from alef.models.model_factory import ModelFactory
from alef.models.mogp_model import MOGPModel
from alef.configs.kernels.hhk_configs import HHKEightLocalDefaultConfig
from alef.configs.kernels.multi_output_kernels.coregionalization_kernel_configs import BasicCoregionalizationMOConfig
from alef.kernels.hierarchical_hyperplane_kernel import HierarchicalHyperplaneKernel
from alef.configs.models.gp_model_config import BasicGPModelConfig
from alef.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from alef.configs.models.gp_model_marginalized_config import (
    BasicGPModelMarginalizedConfig,
    GPModelMarginalizedConfigMoreSamplesConfig,
    GPModelMarginalizedConfigMAPInitialized,
)
from alef.configs.models.mogp_model_config import BasicMOGPModelConfig
from alef.configs.models.sparse_gp_model_config import BasicSparseGPModelConfig
import pytest
import numpy as np


@pytest.mark.parametrize("input_dim,observation_noise", [(1, 0.01), (2, 0.1), (3, 0.2)])
def test_gp_model_building(input_dim, observation_noise):
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=input_dim)
    config = BasicGPModelConfig(kernel_config=kernel_config, observation_noise=observation_noise)
    gp_model = ModelFactory.build(config)
    assert gp_model.observation_noise == config.observation_noise
    assert isinstance(gp_model.kernel, HierarchicalHyperplaneKernel)


@pytest.mark.parametrize("input_dim,observation_noise,n_inducing", [(1, 0.01, 100), (2, 0.1, 200), (3, 0.2, 300)])
def test_sparse_gp_model_building(input_dim, observation_noise, n_inducing):
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=input_dim)
    config = BasicSparseGPModelConfig(
        kernel_config=kernel_config, observation_noise=observation_noise, n_inducing_points=n_inducing
    )
    gp_model = ModelFactory.build(config)
    assert gp_model.observation_noise == config.observation_noise
    assert isinstance(gp_model.kernel, HierarchicalHyperplaneKernel)
    assert gp_model.n_inducing_points == n_inducing


@pytest.mark.parametrize(
    "input_dim,observation_noise,config_class",
    [
        (1, 0.01, GPModelMarginalizedConfigMoreSamplesConfig),
        (2, 0.1, BasicGPModelMarginalizedConfig),
        (3, 0.2, GPModelMarginalizedConfigMAPInitialized),
    ],
)
def test_gp_model_marg_building(input_dim, observation_noise, config_class):
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=input_dim)
    config = config_class(
        kernel_config=kernel_config, observation_noise=observation_noise, expected_observation_noise=observation_noise
    )
    gp_model_marg = ModelFactory.build(config)
    assert gp_model_marg.observation_noise == config.observation_noise
    assert isinstance(gp_model_marg.kernel, HierarchicalHyperplaneKernel)
    x_array = np.random.randn(100, input_dim)
    y_array = np.random.randn(100, 1)
    gp_model_marg.build_model(x_array, y_array)
    prior_rate = gp_model_marg.model.likelihood.variance.prior.rate.numpy()
    should_be_rate = 1.0 / np.power(observation_noise, 2.0)
    assert gp_model_marg.initialization_type == config.initialization_type
    assert gp_model_marg.prediction_quantity == config.prediction_quantity
    if gp_model_marg.train_likelihood_variance:
        assert np.allclose(prior_rate, should_be_rate, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("input_dim,observation_noise", [(1, 0.01), (2, 0.1), (3, 0.2)])
def test_gp_model_laplace_building(input_dim, observation_noise):
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=input_dim)
    config = BasicGPModelLaplaceConfig(
        kernel_config=kernel_config, observation_noise=observation_noise, expected_observation_noise=observation_noise
    )
    gp_model_laplace = ModelFactory.build(config)
    assert gp_model_laplace.observation_noise == config.observation_noise
    assert isinstance(gp_model_laplace.kernel, HierarchicalHyperplaneKernel)
    x_array = np.random.randn(100, input_dim)
    y_array = np.random.randn(100, 1)
    gp_model_laplace.build_model(x_array, y_array)
    prior_rate = gp_model_laplace.model.likelihood.variance.prior.rate.numpy()
    should_be_rate = 1.0 / np.power(observation_noise, 2.0)
    if gp_model_laplace.train_likelihood_variance:
        assert np.allclose(prior_rate, should_be_rate, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("input_dim,output_dim", [(1, 3), (2, 1), (3, 5)])
def test_multi_output_model_building(input_dim, output_dim):
    kernel1_config = BasicCoregionalizationMOConfig(input_dimension=input_dim, output_dimension=output_dim)
    model1_config = BasicMOGPModelConfig(kernel_config=kernel1_config)
    model = ModelFactory.build(model1_config)
    assert isinstance(model, MOGPModel)


def test_deep_gp():
    model_config = DeepGPConfig()
    model = ModelFactory.build(model_config)
    assert isinstance(model, DeepGP)


def test_input_dimension_change():
    kernel_config = BasicCoregionalizationMOConfig(input_dimension=1, output_dimension=1)
    model_config = BasicMOGPModelConfig(kernel_config=kernel_config)
    new_model_config = ModelFactory.change_input_dimension(model_config, 2)
    assert isinstance(new_model_config, BasicMOGPModelConfig)
    assert new_model_config.kernel_config.input_dimension == 2
    assert model_config.kernel_config.input_dimension == 1
    kernel_config = BasicRBFConfig(input_dimension=2)
    model_config = BasicGPModelConfig(kernel_config=kernel_config)
    new_model_config = ModelFactory.change_input_dimension(model_config, 3)
    assert isinstance(new_model_config, BasicGPModelConfig)
    assert new_model_config.kernel_config.input_dimension == 3
    assert model_config.kernel_config.input_dimension == 2


def test_gp_over_kernel_grammar_building():
    kernel_config = OTWeightedDimsExtendedGrammarKernelConfig()
    model_config = BasicObjectGPModelConfig(kernel_config=kernel_config)
    ModelFactory.build(model_config)


if __name__ == "__main__":
    test_gp_over_kernel_grammar_building()
    test_input_dimension_change()
