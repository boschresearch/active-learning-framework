from alef.pools.standard_pool import Pool
from alef.utils.utils import calculate_multioutput_rmse, calculate_rmse
import numpy as np


def test_pool_multivariate_output():
    pool = Pool()
    x_data = np.array([[0, 0], [1, 1], [2, 2]])
    y_data = np.array([[1, 0], [1, 0], [3, 2]])
    pool.set_data(x_data, y_data)
    for i in range(0, x_data.shape[0]):
        y = pool.query(x_data[i])
        assert np.array_equal(y, y_data[i])
        assert y.shape[0] == 2
    pool.set_data(x_data, y_data)
    y = pool.query([0, 0])
    y_data_left_in_pool = pool.get_y_data()
    assert np.array_equal(y_data_left_in_pool[0], y_data[1])


# def test_active_learner_data_sets_multivariate():
#    x_data = np.array([[0, 0], [1, 1], [2, 2]])
#    y_data = np.array([[1, 0], [1, 0], [3, 2]])
#    x_data_initial = np.array([[4, 4], [5, 5], [6, 6]])
#    y_data_initial = np.array([[3, 4], [5, 9], [1, 1]])
#    active_learner_config = ActiveLearningMultiOutputConfig()
#    active_learner_config.validation_type = ValidationType.NEG_LOG_LIKELI
#    active_learner = ActiveLearnerFactory.build(active_learner_config)
#    active_learner.set_pool(x_data, y_data)
#    active_learner.set_initial_dataset_manually(x_data_initial, y_data_initial)
#    query = [0, 0]
#    new_y = active_learner.pool.query(query)
#    active_learner.x_data = np.vstack((active_learner.x_data, query))
#    active_learner.y_data = np.vstack((active_learner.y_data, new_y))
#    assert np.array_equal(np.array([[3, 4], [5, 9], [1, 1], [1, 0]]), active_learner.y_data)


def test_multivariate_rmse():
    y_data = np.array([[1, 0], [1, 0], [3, 2]])
    pred_y = np.array([[0.9, 0.1], [1.1, 0.2], [3.1, 2.15]])
    single_y = np.array([[1], [1], [3]])
    single_pred_y = np.array([[0.9], [1.1], [3.1]])
    rmses = calculate_multioutput_rmse(pred_y, y_data)
    print(rmses)
    assert len(rmses) == y_data.shape[1]
    single_rmse = calculate_rmse(single_pred_y, single_y)
    assert single_rmse == rmses[0]
