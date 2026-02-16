import numpy as np
import pytest
from alef.oracles import (
    BraninHoo,
    Eggholder,
    Exponential2DExtended,
    Exponential2D,
    Hartmann3,
    Hartmann6,
    Sinus,
    OracleNormalizer,
)


@pytest.mark.parametrize(
    "oracle_class", [BraninHoo, Eggholder, Exponential2DExtended, Exponential2D, Hartmann3, Hartmann6, Sinus]
)
def test_standard_oracles(oracle_class):
    oracle = oracle_class(observation_noise=0.01)

    a, b = oracle.get_box_bounds()
    D = oracle.get_dimension()
    X, Y = oracle.get_random_data(1, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == D
    assert Y.shape[0] == 1
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y = oracle.get_random_data_in_box(10, a_set, length * 0.4, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)


def test_oracle_normalizer():
    oracle = OracleNormalizer(BraninHoo(observation_noise=0.1))
    oracle.set_normalization_manually(2.0, 3.0)
    mu, std = oracle.get_normalization()
    assert mu == 2
    assert std == 3
    oracle.set_normalization_by_sampling()

    a, b = oracle.get_box_bounds()
    D = oracle.get_dimension()
    X, Y = oracle.get_random_data(1, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == D
    assert Y.shape[0] == 1
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y = oracle.get_random_data_in_box(10, a_set, length * 0.4, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)


if __name__ == "__main__":
    test_standard_oracles(BraninHoo)
