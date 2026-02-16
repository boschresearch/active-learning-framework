from alef.utils.utils import normalize_data, min_max_normalize_data
from alef.utils.utils import manhatten_distance
from alef.utils.utils import row_wise_compare, row_wise_unique
from alef.utils.utils import check1Dlist
from alef.utils.utils import filter_nan
from alef.utils.utils import create_grid, create_grid_multi_bounds
from alef.utils.utils import tf_delta
import pytest
import numpy as np
import tensorflow as tf
from alef.utils.utils import get_gpytorch_exponential_prior
from tensorflow_probability import distributions as tfd


def test_create_grid_multi_bounds():
    grid = create_grid_multi_bounds([-2, -4, 0, 2], [0, -2, 2, 4], [5, 3, 5, 3])
    result = np.array(
        [
            [-2.0, -4, 0, 2],
            [-1.5, -4, 0, 2],
            [-1.0, -4, 0, 2],
            [-0.5, -4, 0, 2],
            [0.0, -4, 0, 2],
            [-2.0, -3, 0, 2],
            [-1.5, -3, 0, 2],
            [-1.0, -3, 0, 2],
            [-0.5, -3, 0, 2],
            [0.0, -3, 0, 2],
            [-2.0, -2, 0, 2],
            [-1.5, -2, 0, 2],
            [-1.0, -2, 0, 2],
            [-0.5, -2, 0, 2],
            [0.0, -2, 0, 2],
            [-2.0, -4, 0.5, 2],
            [-1.5, -4, 0.5, 2],
            [-1.0, -4, 0.5, 2],
            [-0.5, -4, 0.5, 2],
            [0.0, -4, 0.5, 2],
            [-2.0, -3, 0.5, 2],
            [-1.5, -3, 0.5, 2],
            [-1.0, -3, 0.5, 2],
            [-0.5, -3, 0.5, 2],
            [0.0, -3, 0.5, 2],
            [-2.0, -2, 0.5, 2],
            [-1.5, -2, 0.5, 2],
            [-1.0, -2, 0.5, 2],
            [-0.5, -2, 0.5, 2],
            [0.0, -2, 0.5, 2],
            [-2.0, -4, 1, 2],
            [-1.5, -4, 1, 2],
            [-1.0, -4, 1, 2],
            [-0.5, -4, 1, 2],
            [0.0, -4, 1, 2],
            [-2.0, -3, 1, 2],
            [-1.5, -3, 1, 2],
            [-1.0, -3, 1, 2],
            [-0.5, -3, 1, 2],
            [0.0, -3, 1, 2],
            [-2.0, -2, 1, 2],
            [-1.5, -2, 1, 2],
            [-1.0, -2, 1, 2],
            [-0.5, -2, 1, 2],
            [0.0, -2, 1, 2],
            [-2.0, -4, 1.5, 2],
            [-1.5, -4, 1.5, 2],
            [-1.0, -4, 1.5, 2],
            [-0.5, -4, 1.5, 2],
            [0.0, -4, 1.5, 2],
            [-2.0, -3, 1.5, 2],
            [-1.5, -3, 1.5, 2],
            [-1.0, -3, 1.5, 2],
            [-0.5, -3, 1.5, 2],
            [0.0, -3, 1.5, 2],
            [-2.0, -2, 1.5, 2],
            [-1.5, -2, 1.5, 2],
            [-1.0, -2, 1.5, 2],
            [-0.5, -2, 1.5, 2],
            [0.0, -2, 1.5, 2],
            [-2.0, -4, 2, 2],
            [-1.5, -4, 2, 2],
            [-1.0, -4, 2, 2],
            [-0.5, -4, 2, 2],
            [0.0, -4, 2, 2],
            [-2.0, -3, 2, 2],
            [-1.5, -3, 2, 2],
            [-1.0, -3, 2, 2],
            [-0.5, -3, 2, 2],
            [0.0, -3, 2, 2],
            [-2.0, -2, 2, 2],
            [-1.5, -2, 2, 2],
            [-1.0, -2, 2, 2],
            [-0.5, -2, 2, 2],
            [0.0, -2, 2, 2],
            [-2.0, -4, 0, 3],
            [-1.5, -4, 0, 3],
            [-1.0, -4, 0, 3],
            [-0.5, -4, 0, 3],
            [0.0, -4, 0, 3],
            [-2.0, -3, 0, 3],
            [-1.5, -3, 0, 3],
            [-1.0, -3, 0, 3],
            [-0.5, -3, 0, 3],
            [0.0, -3, 0, 3],
            [-2.0, -2, 0, 3],
            [-1.5, -2, 0, 3],
            [-1.0, -2, 0, 3],
            [-0.5, -2, 0, 3],
            [0.0, -2, 0, 3],
            [-2.0, -4, 0.5, 3],
            [-1.5, -4, 0.5, 3],
            [-1.0, -4, 0.5, 3],
            [-0.5, -4, 0.5, 3],
            [0.0, -4, 0.5, 3],
            [-2.0, -3, 0.5, 3],
            [-1.5, -3, 0.5, 3],
            [-1.0, -3, 0.5, 3],
            [-0.5, -3, 0.5, 3],
            [0.0, -3, 0.5, 3],
            [-2.0, -2, 0.5, 3],
            [-1.5, -2, 0.5, 3],
            [-1.0, -2, 0.5, 3],
            [-0.5, -2, 0.5, 3],
            [0.0, -2, 0.5, 3],
            [-2.0, -4, 1, 3],
            [-1.5, -4, 1, 3],
            [-1.0, -4, 1, 3],
            [-0.5, -4, 1, 3],
            [0.0, -4, 1, 3],
            [-2.0, -3, 1, 3],
            [-1.5, -3, 1, 3],
            [-1.0, -3, 1, 3],
            [-0.5, -3, 1, 3],
            [0.0, -3, 1, 3],
            [-2.0, -2, 1, 3],
            [-1.5, -2, 1, 3],
            [-1.0, -2, 1, 3],
            [-0.5, -2, 1, 3],
            [0.0, -2, 1, 3],
            [-2.0, -4, 1.5, 3],
            [-1.5, -4, 1.5, 3],
            [-1.0, -4, 1.5, 3],
            [-0.5, -4, 1.5, 3],
            [0.0, -4, 1.5, 3],
            [-2.0, -3, 1.5, 3],
            [-1.5, -3, 1.5, 3],
            [-1.0, -3, 1.5, 3],
            [-0.5, -3, 1.5, 3],
            [0.0, -3, 1.5, 3],
            [-2.0, -2, 1.5, 3],
            [-1.5, -2, 1.5, 3],
            [-1.0, -2, 1.5, 3],
            [-0.5, -2, 1.5, 3],
            [0.0, -2, 1.5, 3],
            [-2.0, -4, 2, 3],
            [-1.5, -4, 2, 3],
            [-1.0, -4, 2, 3],
            [-0.5, -4, 2, 3],
            [0.0, -4, 2, 3],
            [-2.0, -3, 2, 3],
            [-1.5, -3, 2, 3],
            [-1.0, -3, 2, 3],
            [-0.5, -3, 2, 3],
            [0.0, -3, 2, 3],
            [-2.0, -2, 2, 3],
            [-1.5, -2, 2, 3],
            [-1.0, -2, 2, 3],
            [-0.5, -2, 2, 3],
            [0.0, -2, 2, 3],
            [-2.0, -4, 0, 4],
            [-1.5, -4, 0, 4],
            [-1.0, -4, 0, 4],
            [-0.5, -4, 0, 4],
            [0.0, -4, 0, 4],
            [-2.0, -3, 0, 4],
            [-1.5, -3, 0, 4],
            [-1.0, -3, 0, 4],
            [-0.5, -3, 0, 4],
            [0.0, -3, 0, 4],
            [-2.0, -2, 0, 4],
            [-1.5, -2, 0, 4],
            [-1.0, -2, 0, 4],
            [-0.5, -2, 0, 4],
            [0.0, -2, 0, 4],
            [-2.0, -4, 0.5, 4],
            [-1.5, -4, 0.5, 4],
            [-1.0, -4, 0.5, 4],
            [-0.5, -4, 0.5, 4],
            [0.0, -4, 0.5, 4],
            [-2.0, -3, 0.5, 4],
            [-1.5, -3, 0.5, 4],
            [-1.0, -3, 0.5, 4],
            [-0.5, -3, 0.5, 4],
            [0.0, -3, 0.5, 4],
            [-2.0, -2, 0.5, 4],
            [-1.5, -2, 0.5, 4],
            [-1.0, -2, 0.5, 4],
            [-0.5, -2, 0.5, 4],
            [0.0, -2, 0.5, 4],
            [-2.0, -4, 1, 4],
            [-1.5, -4, 1, 4],
            [-1.0, -4, 1, 4],
            [-0.5, -4, 1, 4],
            [0.0, -4, 1, 4],
            [-2.0, -3, 1, 4],
            [-1.5, -3, 1, 4],
            [-1.0, -3, 1, 4],
            [-0.5, -3, 1, 4],
            [0.0, -3, 1, 4],
            [-2.0, -2, 1, 4],
            [-1.5, -2, 1, 4],
            [-1.0, -2, 1, 4],
            [-0.5, -2, 1, 4],
            [0.0, -2, 1, 4],
            [-2.0, -4, 1.5, 4],
            [-1.5, -4, 1.5, 4],
            [-1.0, -4, 1.5, 4],
            [-0.5, -4, 1.5, 4],
            [0.0, -4, 1.5, 4],
            [-2.0, -3, 1.5, 4],
            [-1.5, -3, 1.5, 4],
            [-1.0, -3, 1.5, 4],
            [-0.5, -3, 1.5, 4],
            [0.0, -3, 1.5, 4],
            [-2.0, -2, 1.5, 4],
            [-1.5, -2, 1.5, 4],
            [-1.0, -2, 1.5, 4],
            [-0.5, -2, 1.5, 4],
            [0.0, -2, 1.5, 4],
            [-2.0, -4, 2, 4],
            [-1.5, -4, 2, 4],
            [-1.0, -4, 2, 4],
            [-0.5, -4, 2, 4],
            [0.0, -4, 2, 4],
            [-2.0, -3, 2, 4],
            [-1.5, -3, 2, 4],
            [-1.0, -3, 2, 4],
            [-0.5, -3, 2, 4],
            [0.0, -3, 2, 4],
            [-2.0, -2, 2, 4],
            [-1.5, -2, 2, 4],
            [-1.0, -2, 2, 4],
            [-0.5, -2, 2, 4],
            [0.0, -2, 2, 4],
        ]
    )
    assert np.all(grid == result)


def test_create_grid():
    grid = create_grid(-1, 1, 3, 2)
    result = np.array(
        [
            [-1, -1],
            [0, -1],
            [1, -1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
        ]
    )
    assert np.all(grid == result)


def test_normalization():
    x1 = np.random.randn(100, 1) * 10 + 100
    x2 = np.random.randn(100, 1) * 5 - 10
    X = np.concatenate((x1, x2), axis=1)
    assert len(X.shape) == 2
    assert X.shape[1] == 2
    X_normalized = normalize_data(X)
    assert np.isclose(np.mean(X_normalized[:, 0]), 0.0)
    assert np.isclose(np.std(X_normalized[:, 0]), 1.0)
    assert np.isclose(np.mean(X_normalized[:, 1]), 0.0)
    assert np.isclose(np.std(X_normalized[:, 1]), 1.0)
    X = np.random.randn(50, 5) * 10 - 3.0
    X_normalized = normalize_data(X)
    assert np.allclose(np.mean(X_normalized, axis=0), np.repeat(0.0, 5))
    assert np.allclose(np.std(X_normalized, axis=0), np.repeat(1.0, 5))


def test_min_max_normalization():
    x1 = np.random.randn(100, 1) * 10 + 100
    x2 = np.random.randn(100, 1) * 5 - 10
    X = np.concatenate((x1, x2), axis=1)
    assert len(X.shape) == 2
    assert X.shape[1] == 2
    X_normalized = min_max_normalize_data(X)
    assert np.isclose(np.min(X_normalized[:, 0]), 0.0)
    assert np.isclose(np.max(X_normalized[:, 0]), 1.0)
    assert np.isclose(np.min(X_normalized[:, 1]), 0.0)
    assert np.isclose(np.max(X_normalized[:, 1]), 1.0)
    assert np.argmax(X_normalized[:, 0]) == np.argmax(x1)
    assert np.argmax(X_normalized[:, 1]) == np.argmax(x2)


def test_manhatten_distance():
    X = np.array([[0, 0], [1, 1], [1, 2]])
    X2 = np.array([[0, 1], [1, 1], [2, 1]])
    results = np.array([[1, 2, 3], [1, 0, 1], [2, 1, 2]])
    assert np.allclose(results, manhatten_distance(X, X2))


def test_row_wise_compare():
    x1 = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.03], [3.01, 3.02, 3.03]])
    x2 = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.01]])
    np.array([True, False, False])
    assert np.allclose(row_wise_compare(x1, x2), [True, False, False])


def test_row_wise_unique():
    x = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.03], [1.01, 1.02, 1.03]])

    result_x = np.array([[1.01, 1.02, 1.03], [2.01, 2.02, 2.03]])
    result_idx = np.array([0, 1])

    x_uniq, idx = row_wise_unique(x)
    assert np.allclose(x_uniq, result_x)
    assert np.allclose(idx, result_idx)


def test_check1Dlist():
    assert np.all(check1Dlist(1.0, 2) == [1.0, 1.0])
    assert np.all(check1Dlist([1.0], 2) == [1.0, 1.0])
    assert np.all(check1Dlist([1.0, 2.0], 2) == [1.0, 2.0])


def test_filter_nan():
    X = np.arange(16, dtype=float).reshape([8, 2])
    y = np.arange(8, dtype=float).reshape([8, 1])
    y[4, 0] = np.nan
    xx, yy = filter_nan(X, y)

    assert np.all(xx == X[[0, 1, 2, 3, 5, 6, 7]])
    assert np.all(yy == y[[0, 1, 2, 3, 5, 6, 7]])


@pytest.mark.parametrize("lambda_param,x", [(2.0, 1.0), (5.0, 4.0), (10.0, 0.5)])
def test_gyptorch_exponential_prior(lambda_param, x):
    exponential_prior = get_gpytorch_exponential_prior(lambda_param)
    exponential_prior_tf = tfd.Exponential(lambda_param)
    log_p_tf = exponential_prior_tf.log_prob(x).numpy()
    log_p = exponential_prior.log_prob(x).numpy()
    assert np.allclose(log_p, log_p_tf)


def test_tf_delta():
    X = tf.constant([[1, 0], [0, 1], [1, 0]])
    X2 = tf.constant([[1, 1], [1, 0], [0, 0], [0, 1]])
    assert np.all(
        tf.math.equal(
            tf_delta(X, X2),
            tf.constant(
                [[False, True, False, False], [False, False, False, True], [False, True, False, False]], dtype=tf.bool
            ),
        )
    )


if __name__ == "__main__":
    test_filter_nan()
    test_create_grid_multi_bounds()
    test_create_grid()
    test_tf_delta()
    print("Hallo")
