import logging
import torch
from alef.configs.base_parameters import (
    BASE_KERNEL_VARIANCE,
    BASE_KERNEL_LENGTHSCALE,
)
from alef.configs.kernels.change_hyperplane_kernel_config import BasicCHConfig
from alef.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import CKSWithRQGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.compositional_kernel_search_configs import (
    CompositionalKernelSearchGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import (
    NDimFullKernelsGrammarGeneratorConfig,
)
from alef.configs.kernels.periodic_configs import BasicPeriodicConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicLinearKernelPytorchConfig,
    BasicPeriodicKernelPytorchConfig,
    BasicRBFPytorchConfig,
    LinearWithPriorPytorchConfig,
    PeriodicWithPriorPytorchConfig,
    RBFWithPriorPytorchConfig,
    BasicMatern32PytorchConfig,
    BasicMatern52PytorchConfig,
)
from alef.configs.kernels.pytorch_kernels.hhk_pytorch_configs import HHKFourLocalDefaultPytorchConfig
from alef.configs.kernels.spectral_mixture_kernel_config import BasicSMKernelConfig
from alef.configs.models.gp_model_config import BasicGPModelConfig
from alef.configs.models.gp_model_pytorch_config import BasicGPModelPytorchConfig
from alef.kernels.change_hyperplane_kernel import ChangeHyperplaneKernel
from alef.kernels.kernel_grammar.generator_factory import GeneratorFactory
from alef.kernels.kernel_grammar.kernel_grammar import (
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
from alef.kernels.kernel_kernel_hellinger import KernelKernelHellinger
from alef.kernels.pytorch_kernels.elementary_kernels_pytorch import (
    RBFKernelPytorch,
)
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from alef.kernels.warped_multi_index_kernel import WarpedMultiIndexKernel
from alef.configs.kernels.hhk_configs import HHKFourLocalDefaultConfig, HHKTwoLocalDefaultConfig
from alef.configs.kernels.wami_configs import BasicWamiConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.kernels.kernel_factory import KernelFactory
from alef.kernels.rbf_kernel import RBFKernel
from alef.configs.kernels.neural_kernel_network_config import BasicNKNConfig
from alef.configs.kernels.additive_kernel_configs import BasicAdditiveKernelConfig
from alef.kernels.additive_kernel import Partition
from alef.configs.kernels.matern52_configs import BasicMatern52Config
from alef.configs.kernels.matern32_configs import BasicMatern32Config
from alef.configs.kernels.linear_configs import BasicLinearConfig, LinearWithPriorConfig
from alef.kernels.linear_kernel import LinearKernel
from alef.configs.kernels.deep_kernels.invertible_resnet_kernel_configs import (
    BasicInvertibleResnetKernelConfig,
    CurlRegularizedIResnetKernelConfig,
)
from alef.configs.kernels.deep_kernels.mlp_deep_kernel_config import (
    BasicMLPDeepKernelConfig,
    MLPWithPriorDeepKernelConfig,
)
from alef.configs.kernels.grammar_tree_kernel_kernel_configs import (
    KernelGrammarSubtreeKernelConfig,
    OTWeightedDimsExtendedGrammarKernelConfig,
    FeatureType,
)
import numpy as np
import gpflow
import pytest
import gpytorch
from alef.models.gp_model_pytorch import GPModelPytorch
from alef.models.model_factory import ModelFactory
from alef.utils.utils import (
    get_gpytorch_kernel_from_expression_and_state_dict,
    get_hp_sample_from_prior_gpytorch_as_state_dict,
    print_gpytorch_parameters,
)

f64 = gpflow.utilities.to_default_float


def test_hhk_kernel():
    kernel_config = HHKFourLocalDefaultConfig(input_dimension=2)
    kernel = KernelFactory.build(kernel_config)
    # kernel = HierarchicalHyperplaneKernel(**kernel_config.dict())
    kernel_evaluated = kernel.K(np.array([[1.1, 0.5], [0.3, 0.5], [2.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, kernel_evaluated.T, rtol=1e-05, atol=1e-08)


def test_ch_kernel():
    kernel_config = HHKTwoLocalDefaultConfig(input_dimension=2)
    kernel = KernelFactory.build(kernel_config)
    ch_kernel_config = BasicCHConfig(input_dimension=2)
    ch_kernel = ChangeHyperplaneKernel(
        kernel_1=kernel.kernel_list[0], kernel_2=kernel.kernel_list[1], **ch_kernel_config.dict()
    )
    kernel_evaluated = kernel.K(np.array([[1.1, 0.5], [0.3, 0.5], [2.0, 1.0]])).numpy()
    ch_kernel_evaluated = ch_kernel.K(np.array([[1.1, 0.5], [0.3, 0.5], [2.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, ch_kernel_evaluated, rtol=1e-05, atol=1e-08)
    kernel_evaluated = kernel.K_diag(np.array([[1.1, 0.5], [0.3, 0.5], [2.0, 1.0]])).numpy()
    ch_kernel_evaluated = ch_kernel.K_diag(np.array([[1.1, 0.5], [0.3, 0.5], [2.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, ch_kernel_evaluated, rtol=1e-05, atol=1e-08)


def test_wami_kernel():
    kernel_config = BasicWamiConfig(input_dimension=2)
    kernel = WarpedMultiIndexKernel(**kernel_config.dict())
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, kernel_evaluated.T, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("lengthscale", (0.5, 1.0, 2.0))
def test_rbf_kernel(lengthscale):
    kernel_config = BasicRBFConfig(input_dimension=2, base_lengthscale=lengthscale)
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    rbf_kernel = gpflow.kernels.RBF(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale, lengthscale])
    rbf_kernel_eval = rbf_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)
    kernel_evaluated = kernel.K(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    rbf_kernel_eval = rbf_kernel(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)
    kernel_evaluated = kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    rbf_kernel_eval = rbf_kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)
    kernel_config = BasicRBFConfig(
        input_dimension=2,
        base_lengthscale=lengthscale,
        active_on_single_dimension=True,
        active_dimension=1,
    )
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    rbf_kernel = gpflow.kernels.RBF(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale], active_dims=[1])
    rbf_kernel_eval = rbf_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, rbf_kernel_eval)


def test_additive_kernel():
    partition1 = Partition(4)
    partition1.add_partition_element([0, 2])
    partition1.add_partition_element([1])
    partition1.add_partition_element([3])
    partition2 = Partition(6)
    partition2.add_partition_element([0, 2, 4])
    partition2.add_partition_element([1])
    partition2.add_partition_element([3, 5])
    kernel_config1 = BasicAdditiveKernelConfig(input_dimension=4, partition=partition1)
    kernel_config2 = BasicAdditiveKernelConfig(input_dimension=6, partition=partition2)
    kernel1 = KernelFactory.build(kernel_config1)
    kernel2 = KernelFactory.build(kernel_config2)
    X1 = np.random.uniform(-5, 5, (7, 4))
    X2 = np.random.uniform(-5, 5, (7, 6))
    kernel_evaluated1 = kernel1.K(X1).numpy()
    assert np.allclose(kernel_evaluated1, kernel_evaluated1.T, rtol=1e-05, atol=1e-08)
    kernel_evaluated2 = kernel2.K(X2).numpy()
    assert np.allclose(kernel_evaluated2, kernel_evaluated2.T, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize("lengthscale", (0.5, 1.0, 2.0))
def test_matern_kernel(lengthscale):
    kernel_config = BasicMatern52Config(input_dimension=2, base_lengthscale=lengthscale)
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    matern_kernel = gpflow.kernels.Matern52(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale, lengthscale])
    matern_kernel_eval = matern_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)
    kernel_evaluated = kernel.K(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    matern_kernel_eval = matern_kernel(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)
    kernel_evaluated = kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    matern_kernel_eval = matern_kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)
    kernel_config = BasicMatern52Config(
        input_dimension=2,
        base_lengthscale=lengthscale,
        active_on_single_dimension=True,
        active_dimension=1,
    )
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    matern_kernel = gpflow.kernels.Matern52(variance=BASE_KERNEL_VARIANCE, lengthscales=[lengthscale], active_dims=[1])
    matern_kernel_eval = matern_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, matern_kernel_eval)


@pytest.mark.parametrize("variance", (0.5, 1.0, 2.0))
def test_linear_kernel(variance):
    kernel_config = BasicLinearConfig(input_dimension=2, base_variance=variance)
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    linear_kernel = gpflow.kernels.Polynomial(degree=1, variance=variance, offset=kernel_config.base_offset)
    linear_kernel_eval = linear_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, linear_kernel_eval)
    kernel_evaluated = kernel.K(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    linear_kernel_eval = linear_kernel(
        np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]]),
        np.array([[1.0, 0.8], [0.1, 0.2], [0.0, 1.0]]),
    ).numpy()
    assert np.allclose(kernel_evaluated, linear_kernel_eval)
    kernel_evaluated = kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    linear_kernel_eval = linear_kernel.K_diag(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, linear_kernel_eval)
    kernel_config = BasicLinearConfig(
        input_dimension=3,
        base_variance=variance,
        active_on_single_dimension=True,
        active_dimension=2,
    )
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5, 0.1], [0.1, 0.2, 0.3], [0.0, 1.0, 0.4]])).numpy()
    linear_kernel = gpflow.kernels.Polynomial(
        degree=1, variance=variance, offset=kernel_config.base_offset, active_dims=[2]
    )
    linear_kernel_eval = linear_kernel(np.array([[1.0, 0.5, 0.1], [0.1, 0.2, 0.3], [0.0, 1.0, 0.4]])).numpy()
    assert np.allclose(kernel_evaluated, linear_kernel_eval)


@pytest.mark.parametrize("period", (0.5, 1.0, 2.0))
def test_periodic_kernel(period):
    kernel_config = BasicPeriodicConfig(input_dimension=2, base_period=period)
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    periodic_kernel = gpflow.kernels.Periodic(
        gpflow.kernels.RBF(
            lengthscales=f64(np.repeat(BASE_KERNEL_LENGTHSCALE, 2)),
            variance=f64([BASE_KERNEL_VARIANCE]),
        ),
        period=f64([period]),
    )
    periodic_kernel_eval = periodic_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, periodic_kernel_eval)
    kernel_config = BasicPeriodicConfig(
        input_dimension=2,
        base_period=period,
        active_on_single_dimension=True,
        active_dimension=0,
    )
    kernel = KernelFactory.build(kernel_config)
    kernel_evaluated = kernel.K(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    periodic_kernel = gpflow.kernels.Periodic(
        gpflow.kernels.RBF(
            lengthscales=f64(np.repeat(BASE_KERNEL_LENGTHSCALE, 1)),
            variance=f64([BASE_KERNEL_VARIANCE]),
            active_dims=[0],
        ),
        period=f64([period]),
    )
    periodic_kernel_eval = periodic_kernel(np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])).numpy()
    assert np.allclose(kernel_evaluated, periodic_kernel_eval)


def test_deep_kernel():
    kernel_config1 = BasicMLPDeepKernelConfig(input_dimension=4)
    kernel_config2 = MLPWithPriorDeepKernelConfig(input_dimension=6)
    kernel1 = KernelFactory.build(kernel_config1)
    kernel2 = KernelFactory.build(kernel_config2)
    X1 = np.random.uniform(-5, 5, (7, 4))
    X2 = np.random.uniform(-5, 5, (7, 6))
    kernel_evaluated1 = kernel1.K(X1).numpy()
    assert np.allclose(kernel_evaluated1, kernel_evaluated1.T, rtol=1e-05, atol=1e-08)
    kernel_evaluated2 = kernel2.K(X2).numpy()
    assert np.allclose(kernel_evaluated2, kernel_evaluated2.T, rtol=1e-05, atol=1e-08)


def test_invertible_resnet_kernel():
    kernel_config1 = BasicInvertibleResnetKernelConfig(input_dimension=1)
    kernel_config2 = BasicInvertibleResnetKernelConfig(input_dimension=6)
    kernel_config3 = CurlRegularizedIResnetKernelConfig(input_dimension=6)
    kernel1 = KernelFactory.build(kernel_config1)
    kernel2 = KernelFactory.build(kernel_config2)
    kernel3 = KernelFactory.build(kernel_config3)
    X1 = np.random.uniform(-5, 5, (7, 1))
    X2 = np.random.uniform(-5, 5, (7, 6))
    kernel_evaluated1 = kernel1.K(X1).numpy()
    assert np.allclose(kernel_evaluated1, kernel_evaluated1.T, rtol=1e-05, atol=1e-08)
    kernel_evaluated2 = kernel2.K(X2).numpy()
    assert np.allclose(kernel_evaluated2, kernel_evaluated2.T, rtol=1e-05, atol=1e-08)
    kernel_evaluated3 = kernel3.K(X2).numpy()
    assert np.allclose(kernel_evaluated3, kernel_evaluated3.T, rtol=1e-05, atol=1e-08)


def test_neural_kernel_network():
    kernel_config1 = BasicNKNConfig(input_dimension=1)
    kernel_config2 = BasicNKNConfig(input_dimension=6)
    kernel1 = KernelFactory.build(kernel_config1)
    kernel2 = KernelFactory.build(kernel_config2)
    X1 = np.random.uniform(-5, 5, (7, 1))
    X2 = np.random.uniform(-5, 5, (7, 6))
    kernel_evaluated1 = kernel1.K(X1).numpy()
    assert np.allclose(kernel_evaluated1, kernel_evaluated1.T, rtol=1e-05, atol=1e-08)
    kernel_evaluated2 = kernel2.K(X2).numpy()
    assert np.allclose(kernel_evaluated2, kernel_evaluated2.T, rtol=1e-05, atol=1e-08)


def test_kernel_kernel_grammar_tree():
    kernel_config = KernelGrammarSubtreeKernelConfig()
    kernel_kernel = KernelFactory.build(kernel_config)
    kernel_grammar_generator = GeneratorFactory.build(NDimFullKernelsGrammarGeneratorConfig(input_dimension=2))
    kernel_grammar_expression_list = kernel_grammar_generator.get_random_canditates(
        kernel_grammar_generator.search_space.get_num_base_kernels() * 3
    )
    K1 = kernel_kernel.K(kernel_kernel.transform_X(kernel_grammar_expression_list)).numpy()
    kernel_grammar_expression_list2 = kernel_grammar_generator.get_random_canditates(
        kernel_grammar_generator.search_space.get_num_base_kernels() * 5
    )
    K2 = kernel_kernel.K(
        kernel_kernel.transform_X(kernel_grammar_expression_list),
        kernel_kernel.transform_X(kernel_grammar_expression_list2),
    ).numpy()
    K3 = kernel_kernel.K(
        kernel_kernel.transform_X(kernel_grammar_expression_list2),
        kernel_kernel.transform_X(kernel_grammar_expression_list),
    ).numpy()
    assert np.allclose(K1, K1.T, rtol=1e-05, atol=1e-08)
    assert np.allclose(K2, K3.T, rtol=1e-05, atol=1e-08)


def test_hellinger_kernel_kernel_caching():
    base_expression_1 = ElementaryKernelGrammarExpression(RBFKernel(**RBFWithPriorConfig(input_dimension=2).dict()))
    base_expression_2 = ElementaryKernelGrammarExpression(
        LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict())
    )
    expression = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.MULTIPLY)
    expression3 = KernelGrammarExpression(expression, base_expression_2, KernelGrammarOperator.MULTIPLY)
    expressions = [base_expression_1, base_expression_2, expression, expression2]
    expressions2 = [base_expression_1, expression3]
    kernel_kernel = KernelKernelHellinger(2, 1.0, 1.0, 20, 10, True, True, (0.1, 0.7), (0.4, 0.7))
    assert kernel_kernel.evaluated_kernel_cache.get_number_of_cached_kernels() == 0
    K = kernel_kernel.K(kernel_kernel.transform_X(expressions))
    assert kernel_kernel.evaluated_kernel_cache.get_number_of_cached_kernels() == len(expressions)
    assert len(kernel_kernel.expected_distance_cache.cache_dict) > 0
    K2 = kernel_kernel.K(kernel_kernel.transform_X(expressions))
    assert kernel_kernel.evaluated_kernel_cache.get_number_of_cached_kernels() == len(expressions)
    assert len(kernel_kernel.expected_distance_cache.cache_dict) > 0
    assert np.allclose(K, K2, rtol=1e-05, atol=1e-08)
    K3 = kernel_kernel.K(kernel_kernel.transform_X(expressions), kernel_kernel.transform_X(expressions2))
    assert kernel_kernel.evaluated_kernel_cache.get_number_of_cached_kernels() == len(expressions) + 1
    assert len(kernel_kernel.expected_distance_cache.cache_dict) > 0
    K4 = kernel_kernel.K(kernel_kernel.transform_X(expressions), kernel_kernel.transform_X(expressions2))
    assert np.allclose(K3, K4, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize(
    "dim,generator_config_class",
    [
        (1, CKSWithRQGeneratorConfig),
        (2, CKSWithRQGeneratorConfig),
        (3, CompositionalKernelSearchGeneratorConfig),
        (4, CompositionalKernelSearchGeneratorConfig),
    ],
)
def test_ot_kernel(dim, generator_config_class):
    kernel_config = OTWeightedDimsExtendedGrammarKernelConfig(input_dimension=dim)
    kernel_kernel = KernelFactory.build(kernel_config)
    kernel_grammar_generator = GeneratorFactory.build(generator_config_class(input_dimension=dim))
    kernel_grammar_expression_list = kernel_grammar_generator.get_random_canditates(
        kernel_grammar_generator.search_space.get_num_base_kernels() * 3
    )
    K1 = kernel_kernel.K(kernel_kernel.transform_X(kernel_grammar_expression_list)).numpy()
    kernel_grammar_expression_list2 = kernel_grammar_generator.get_random_canditates(
        kernel_grammar_generator.search_space.get_num_base_kernels() * 5
    )
    K2 = kernel_kernel.K(
        kernel_kernel.transform_X(kernel_grammar_expression_list),
        kernel_kernel.transform_X(kernel_grammar_expression_list2),
    ).numpy()
    K3 = kernel_kernel.K(
        kernel_kernel.transform_X(kernel_grammar_expression_list2),
        kernel_kernel.transform_X(kernel_grammar_expression_list),
    ).numpy()
    assert np.allclose(K1, K1.T, rtol=1e-05, atol=1e-08)
    assert np.allclose(K2, K3.T, rtol=1e-05, atol=1e-08)
    assert np.std(K1) > 0.0
    assert np.std(K2) > 0.0
    assert np.std(K3) > 0.0
    feature_dicts = kernel_kernel.internal_transform_X(
        kernel_grammar_expression_list, FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT
    )
    for feature_dict in feature_dicts:
        value_sum = 0.0
        for key in feature_dict:
            value_sum += feature_dict[key]
        assert np.allclose(value_sum, float(dim))


def test_spectral_mixture_kernel():
    kernel = KernelFactory.build(BasicSMKernelConfig(input_dimension=2, num_mixtures=2))
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    means = np.array([[0.5, 0.3], [0.5, 0.3]])
    weights = np.array([0.5, 0.5])
    scales = np.array([[1.0, 1.0], [1.0, 1.0]])
    kernel.set_parameters(weights, means, scales)
    K = kernel(X)
    diag_should_be = np.cos(2 * np.pi * -0.2) * np.power(np.exp(-2 * np.power(np.pi, 2.0)), 2.0)
    assert np.allclose(K[0, 1], diag_should_be)
    X1 = np.random.uniform(-5, 5, (7, 2))
    X2 = np.random.uniform(-5, 5, (7, 2))
    kernel_evaluated1 = kernel.K(X1).numpy()
    assert np.allclose(kernel_evaluated1, kernel_evaluated1.T, rtol=1e-05, atol=1e-08)
    kernel_evaluated2 = kernel.K(X2).numpy()
    assert np.allclose(kernel_evaluated2, kernel_evaluated2.T, rtol=1e-05, atol=1e-08)
    np.linalg.cholesky(kernel_evaluated1)


@pytest.mark.parametrize(
    "input_dim,pytorch_kernel_config,gpflow_kernel_config",
    [
        (1, BasicRBFPytorchConfig, BasicRBFConfig),
        (3, BasicRBFPytorchConfig, BasicRBFConfig),
        (2, BasicPeriodicKernelPytorchConfig, BasicPeriodicConfig),
        (3, BasicLinearKernelPytorchConfig, BasicLinearConfig),
        (2, BasicMatern52PytorchConfig, BasicMatern52Config),
        (3, BasicMatern32PytorchConfig, BasicMatern32Config),
    ],
)
def test_pytorch_kernel(input_dim, pytorch_kernel_config, gpflow_kernel_config):
    pytorch_kernel = PytorchKernelFactory.build(pytorch_kernel_config(input_dimension=input_dim, base_variance=0.7))
    gpflow_kernel = KernelFactory.build(gpflow_kernel_config(input_dimension=input_dim, base_variance=0.7))
    print_gpytorch_parameters(pytorch_kernel)
    X1 = np.random.uniform(-5, 5, (7, input_dim))
    np.random.uniform(-5, 5, (7, input_dim))
    K_pt = pytorch_kernel(torch.tensor(X1)).numpy()
    K_gpf = gpflow_kernel(X1).numpy()
    assert np.allclose(K_pt, K_gpf, rtol=1e-05, atol=1e-08)


def test_additive_kernel_wrapper_gpytorch():
    X = np.random.uniform(0.0, 1.0, (10, 2))
    X_torch = torch.from_numpy(X)
    base_expression1 = ElementaryKernelGrammarExpression(
        RBFKernelPytorch(
            **RBFWithPriorPytorchConfig(input_dimension=2, active_on_single_dimension=True, active_dimension=0).dict()
        )
    )
    kernel1 = base_expression1.get_kernel()
    kernel2 = gpytorch.kernels.AdditiveKernel(kernel1)
    assert np.allclose(kernel1(X_torch).numpy(), kernel2(X_torch).numpy())


def test_kernel_grammar_and_torch_state_dict_interplay():
    X = np.random.uniform(0.0, 1.0, (10, 2))
    X_torch = torch.from_numpy(X)
    se_kernel_config = RBFWithPriorPytorchConfig(input_dimension=2, active_on_single_dimension=True, active_dimension=0)
    se_kernel_config_2 = RBFWithPriorPytorchConfig(
        input_dimension=2, active_on_single_dimension=True, active_dimension=1
    )
    lin_kernel_config = LinearWithPriorPytorchConfig(
        input_dimension=2, active_on_single_dimension=True, active_dimension=1
    )
    per_kernel_config = PeriodicWithPriorPytorchConfig(
        input_dimension=2, active_on_single_dimension=True, active_dimension=0
    )
    se_kernel = PytorchKernelFactory.build(se_kernel_config)
    se_kernel2 = PytorchKernelFactory.build(se_kernel_config_2)
    lin_kernel = PytorchKernelFactory.build(lin_kernel_config)
    per_kernel = PytorchKernelFactory.build(per_kernel_config)

    base_expression1 = ElementaryKernelGrammarExpression(se_kernel)
    base_expression2 = ElementaryKernelGrammarExpression(se_kernel2)
    base_expression3 = ElementaryKernelGrammarExpression(lin_kernel)
    base_expression4 = ElementaryKernelGrammarExpression(per_kernel)

    expression1 = KernelGrammarExpression(base_expression1, base_expression4, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(expression1, base_expression3, KernelGrammarOperator.MULTIPLY)
    expression3 = KernelGrammarExpression(expression2, expression1, KernelGrammarOperator.MULTIPLY)
    expression4 = KernelGrammarExpression(base_expression2, expression3, KernelGrammarOperator.ADD)
    expression_list = [base_expression1, expression1, expression2, expression3, expression4]
    for expression in expression_list:
        state_dict = get_hp_sample_from_prior_gpytorch_as_state_dict(expression, True)
        kernel = get_gpytorch_kernel_from_expression_and_state_dict(expression, state_dict, True)
        K1 = kernel(X_torch).numpy()
        expression_copied = expression.deep_copy()
        kernel2 = get_gpytorch_kernel_from_expression_and_state_dict(expression_copied, state_dict, True)
        K2 = kernel2(X_torch).numpy()
        kernel3 = expression.get_kernel()
        kernel4 = gpytorch.kernels.AdditiveKernel(kernel3)
        state_dict4 = kernel4.state_dict()
        kernel5 = get_gpytorch_kernel_from_expression_and_state_dict(expression, state_dict4, True)
        K3 = kernel3(X_torch).numpy()
        K4 = kernel5(X_torch).numpy()
        assert np.allclose(K1, K2)
        assert np.allclose(K3, K4)


def test_gyptorch_hhk():
    torch.set_default_dtype(torch.float64)
    gpytorch_hhk = PytorchKernelFactory.build(HHKFourLocalDefaultPytorchConfig(input_dimension=3))
    gpflow_hhk = KernelFactory.build(HHKFourLocalDefaultConfig(input_dimension=3))
    hyperplanes = np.random.randn(4, 3)
    smoothing = np.random.rand(3)
    gpytorch_hhk.smoothing = torch.from_numpy(smoothing)
    gpytorch_hhk._set_hyperplanes(torch.from_numpy(hyperplanes))
    gpflow_hhk.set_hyperplane_parameters([hyperplanes[:, i] for i in range(0, 3)])
    gpflow_hhk.set_smoothing_list(np.expand_dims(smoothing, axis=1))
    X = np.random.randn(10, 3)
    K_gpflow = gpflow_hhk(X)
    K_pytorch = gpytorch_hhk(torch.from_numpy(X)).numpy()
    assert np.allclose(K_gpflow, K_pytorch)

    X_test = np.random.uniform(0.0, 1.0, size=(10, 3))
    X_data = np.random.uniform(0.0, 1.0, size=(10, 3))
    y_data = np.random.uniform(0.0, 1.0, size=(10, 1))
    kernel_config = BasicRBFConfig(input_dimension=3)
    model_config = BasicGPModelConfig(kernel_config=kernel_config)
    model_config.optimize_hps = False
    model_config.observation_noise = 0.1
    kernel_config_torch = BasicRBFPytorchConfig(input_dimension=3)
    model_config_torch = BasicGPModelPytorchConfig(kernel_config=kernel_config_torch)
    model_config_torch.initial_likelihood_noise = 0.1
    model_config_torch.optimize_hps = False

    model = ModelFactory.build(model_config)
    model.set_kernel(gpflow_hhk)
    # model.kernel = gpflow.kernels.RBF()
    model.infer(X_data, y_data)
    mu_gpflow, sigma_gpflow = model.predictive_dist(X_test)
    model_gpytroch = GPModelPytorch(kernel=gpytorch_hhk, **model_config_torch.dict())
    # model_gpytroch.set_kernel(gpytorch_hhk)
    model_gpytroch.infer(X_data, y_data)
    mu_gpytorch, sigma_gpytorch = model_gpytroch.predictive_dist(X_test)
    assert np.allclose(mu_gpytorch, mu_gpflow, atol=1e-4)
    assert np.allclose(sigma_gpytorch, sigma_gpflow, atol=1e-4)
    torch.set_default_dtype(torch.float32)
    # print(K_pytorch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_gyptorch_hhk()
