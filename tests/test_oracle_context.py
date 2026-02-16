import numpy as np
from alef.oracles.hartmann6 import Hartmann6


def test_get_grid_data():
    oracle = Hartmann6(0.01)
    oracle.set_context([3, 4, 5], [0.5, 0.6, 0.7])
    X, Y = oracle.get_grid_data(10, noisy=False)
    assert X.shape[0] == 1000
    assert np.all(X[:, 3] == 0.5)
    assert np.all(X[:, 4] == 0.6)
    assert np.all(X[:, 5] == 0.7)


def test_get_random_data():
    oracle = Hartmann6(0.01)
    oracle.set_context([3, 4, 5], [0.5, 0.6, 0.7])
    X, Y = oracle.get_random_data(100, noisy=False)
    assert np.all(X[:, 3] == 0.5)
    assert np.all(X[:, 4] == 0.6)
    assert np.all(X[:, 5] == 0.7)


def test_get_random_data_in_box():
    oracle = Hartmann6(0.01)
    oracle.set_context([3, 4, 5], [0.5, 0.6, 0.7])
    X, Y = oracle.get_random_data_in_box(
        100, [0, 0.3, 0.5, -np.inf, -np.inf, -np.inf], [0.2, 0.2, 0.2, 0, 0, 0], noisy=False
    )

    assert np.all(X[:, 0] >= 0)
    assert np.all(X[:, 0] <= 0.2)
    assert np.all(X[:, 1] >= 0.3)
    assert np.all(X[:, 1] <= 0.5)
    assert np.all(X[:, 2] >= 0.5)
    assert np.all(X[:, 2] <= 0.7)

    assert np.all(X[:, 3] == 0.5)
    assert np.all(X[:, 4] == 0.6)
    assert np.all(X[:, 5] == 0.7)


if __name__ == "__main__":
    test_get_grid_data()
    test_get_random_data()
    test_get_random_data_in_box()
