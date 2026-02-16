import pytest
from alef.active_learner.active_learner_factory import ActiveLearnerFactory
from alef.configs.active_learner.active_learner_oracle_configs import (
    PredEntropyActiveLearnerOracleConfig,
    PredVarActiveLearnerOracleConfig,
    RandomActiveLearnerOracleConfig,
)
from alef.configs.active_learner.continuous_policy_active_learner_configs import (
    PytestContinuousPolicyActiveLearnerOracleConfig,
)
from alef.configs.active_learner.active_learner_configs import (
    PredEntropyActiveLearnerConfig,
    PredVarActiveLearnerConfig,
    RandomActiveLearnerConfig,
)
from alef.configs.active_learner.batch_active_learner_configs import (
    EntropyBatchActiveLearnerConfig,
    RandomBatchActiveLearnerConfig,
)
from alef.enums.global_model_enums import PredictionQuantity
from alef.pools.standard_pool import Pool
from alef.pools.pool_from_oracle import PoolFromOracle
import numpy as np
from alef.oracles import (
    GPOracle1D,
    BraninHoo,
    OracleNormalizer,
)
from alef.models.model_factory import ModelFactory
from alef.configs.models.gp_model_config import GPModelFastConfig
from alef.configs.models.gp_model_marginalized_config import BasicGPModelMarginalizedConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.enums.active_learner_enums import (
    ValidationType,
    OracleALAcquisitionOptimizationType,
)
from alef.oracles.exponential_2d import Exponential2D
from alef.active_learner.safe_active_learner import SafeActiveLearner
from alef.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from alef.configs.acquisition_function.safe_acquisition_functions.safe_pred_entropy_config import (
    BasicSafePredEntropyAllConfig,
)


def test_pool():
    pool = Pool()
    x_data = np.array([0, 1, 2])
    y_data = np.array([1, 2, 3])
    pool.set_data(x_data, y_data)
    for i in range(0, x_data.shape[0]):
        y = pool.query(x_data[i])
        assert y == y_data[i]
    pool.set_data(x_data, y_data)
    assert pool.query(1) == 2
    assert x_data.shape[0] == 3


@pytest.mark.parametrize(
    "active_learner_config_class",
    (PredEntropyActiveLearnerConfig, PredVarActiveLearnerConfig, RandomActiveLearnerConfig),
)
def test_standard_active_learner(tmp_path, active_learner_config_class):
    kernel_config = BasicRBFConfig(input_dimension=1)
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    al_config = active_learner_config_class()
    model = ModelFactory.build(model_config)
    gp_oracle = GPOracle1D(kernel_config, 0.01)
    gp_oracle.initialize(0, 1, 1000)
    x_pool, y_pool = gp_oracle.get_random_data(5000)
    x_test, y_test = gp_oracle.get_random_data(40)
    active_learner = ActiveLearnerFactory.build(al_config)
    active_learner.set_model(model)
    active_learner.set_pool(x_pool, y_pool)
    active_learner.set_test_set(x_test, y_test)
    active_learner.sample_initial_data(5)
    # set saving
    active_learner.save_experiment_summary_to_path(tmp_path, "AL_result.xlsx")
    val, x_queries = active_learner.learn(3)
    assert len(val) == 3


def test_active_learner_log_likeli():
    kernel_config = BasicRBFConfig(input_dimension=1)
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    al_config = PredVarActiveLearnerConfig(validation_type=ValidationType.NEG_LOG_LIKELI)
    model = ModelFactory.build(model_config)
    gp_oracle = GPOracle1D(kernel_config, 0.01)
    gp_oracle.initialize(0, 1, 1000)
    x_pool, y_pool = gp_oracle.get_random_data(5000)
    x_test, y_test = gp_oracle.get_random_data(40)
    active_learner = ActiveLearnerFactory.build(al_config)
    active_learner.set_model(model)
    active_learner.set_pool(x_pool, y_pool)
    active_learner.set_test_set(x_test, y_test)
    active_learner.sample_initial_data(5)
    val, x_queries = active_learner.learn(3)
    assert not val.isnull().values.any()
    assert len(val) == 3


@pytest.mark.parametrize(
    "active_learner_config_class", (EntropyBatchActiveLearnerConfig, RandomBatchActiveLearnerConfig)
)
def test_batch_active_learner(tmp_path, active_learner_config_class):
    data_set_size = 5
    batch_size = 5
    n_steps = 3
    kernel_config = RBFWithPriorConfig(input_dimension=1)
    model_config = GPModelFastConfig(
        kernel_config=kernel_config, observation_noise=0.01, prediction_quantity=PredictionQuantity.PREDICT_F
    )
    model = ModelFactory.build(model_config)
    gp_oracle = GPOracle1D(kernel_config, 0.01)
    gp_oracle.initialize(-10, 10, 1000)
    x_pool, y_pool = gp_oracle.get_random_data(100)
    x_test, y_test = gp_oracle.get_random_data(40)
    active_learner = ActiveLearnerFactory.build(active_learner_config_class())
    # active_learner.set_do_plotting(True)
    active_learner.set_model(model)
    active_learner.set_pool(x_pool, y_pool)
    active_learner.set_test_set(x_test, y_test)
    active_learner.sample_initial_data(data_set_size)
    # set saving
    active_learner.save_experiment_summary_to_path(tmp_path, "AL_result.xlsx")
    val, x_queries = active_learner.learn(n_steps)
    print(val)
    assert len(val) == n_steps
    assert len(x_queries) == data_set_size + n_steps * batch_size


@pytest.mark.parametrize(
    "active_learner_config_class",
    (PredEntropyActiveLearnerOracleConfig, PredVarActiveLearnerOracleConfig, RandomActiveLearnerOracleConfig),
)
def test_oracle_active_learner(tmp_path, active_learner_config_class):
    data_set_size = 3
    test_set_size = 200
    n_steps = 3
    oracle = Exponential2D(0.01)
    x_data, y_data = oracle.get_random_data(data_set_size)
    x_test, y_test = oracle.get_random_data(test_set_size)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    active_learner = ActiveLearnerFactory.build(active_learner_config_class())
    # active_learner.set_do_plotting(True)
    active_learner.set_oracle(oracle)
    active_learner.sample_train_set(data_set_size, set_seed=True)
    active_learner.set_test_set(x_data, y_data)
    active_learner.set_model(model)
    # set saving
    active_learner.save_experiment_summary_to_path(tmp_path, "AL_result.xlsx")
    active_learner.learn(n_steps)


def test_oracle_amortized_active_learner(tmp_path):
    data_set_size = 3
    test_set_size = 200
    n_steps = 3
    oracle = BraninHoo(0.01)
    x_data, y_data = oracle.get_random_data(data_set_size)
    x_test, y_test = oracle.get_random_data(test_set_size)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    al_config = PytestContinuousPolicyActiveLearnerOracleConfig(
        validation_at=[0, n_steps - 1], policy_dimension=oracle.get_dimension()
    )
    active_learner = ActiveLearnerFactory.build(al_config)
    # active_learner.set_do_plotting(True)
    active_learner.set_oracle(oracle)
    active_learner.sample_train_set(data_set_size, set_seed=True)
    active_learner.set_test_set(x_data, y_data)
    active_learner.set_model(model)
    # set saving
    active_learner.save_experiment_summary_to_path(tmp_path, "AL_result.xlsx")
    active_learner.learn(n_steps)


def test_oracle_active_learner_marginalized():
    data_set_size = 3
    test_set_size = 200
    n_steps = 1
    # oracle = GPOracle1D(gpflow.kernels.RBF(lengthscales=0.2),0.01)
    # oracle.initialize(0,1,2000)
    oracle = Exponential2D(0.01)
    x_data, y_data = oracle.get_random_data(data_set_size)
    x_test, y_test = oracle.get_random_data(test_set_size)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = BasicGPModelMarginalizedConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    active_learner_config = PredVarActiveLearnerOracleConfig()
    active_learner_config.validation_type = ValidationType.RMSE
    active_learner_config.acquisiton_optimization_type = OracleALAcquisitionOptimizationType.RANDOM_SHOOTING
    active_learner = ActiveLearnerFactory.build(active_learner_config)
    active_learner.set_oracle(oracle)
    active_learner.sample_train_set(data_set_size, set_seed=True)
    active_learner.set_test_set(x_data, y_data)
    active_learner.set_model(model)
    active_learner.learn(n_steps)


def test_safe_active_learner(tmp_path):
    oracle = OracleNormalizer(BraninHoo(0.1))
    oracle.set_normalization_manually(60.088767740805736, 62.34134408167649)
    X, Y = oracle.get_random_data(100, noisy=False)
    Y_max = np.max(Y)
    Y_min = np.min(Y)
    safe_lower = 0.9 * Y_min + 0.1 * Y_max
    safe_upper = 0.1 * Y_min + 0.9 * Y_max

    pool = PoolFromOracle(oracle)
    pool.discretize_random(2000)
    acq_func = AcquisitionFunctionFactory.build(
        BasicSafePredEntropyAllConfig(safety_thresholds_lower=safe_lower, safety_thresholds_upper=safe_upper)
    )
    model = ModelFactory.build(GPModelFastConfig(kernel_config=BasicRBFConfig(input_dimension=oracle.get_dimension())))
    data_init = pool.get_random_data(10, noisy=True)
    data_test = pool.get_random_data(100, noisy=True)

    learner = SafeActiveLearner(
        acq_func,
        ValidationType.RMSE,
        do_plotting=False,
        query_noisy=True,
        model_is_safety_model=True,
        save_results=True,
        experiment_path=tmp_path,
    )
    learner.set_pool(pool)
    learner.set_model(model, safety_models=None)

    # perform the main experiment
    learner.set_train_data(*data_init)
    learner.set_test_data(*data_test)
    _, _, _ = learner.learn(5)


if __name__ == "__main__":
    test_batch_active_learner(RandomBatchActiveLearnerConfig)
