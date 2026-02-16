import numpy as np
import gpflow
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from alef.kernels.multi_output_kernels.coregionalization_1latent_kernel import Coregionalization1LKernel
from alef.kernels.multi_output_kernels.coregionalization_kernel import (
    CoregionalizationMOKernel,
    CoregionalizationSOKernel,
)
from alef.kernels.multi_output_kernels.coregionalization_Platent_kernel import CoregionalizationPLKernel
from alef.kernels.multi_output_kernels.flexible_transfer_kernel import FlexibleTransferKernel
from alef.kernels.multi_output_kernels.multi_source_additive_kernel import MIAdditiveKernel

N = 10
D = 2  # don't change this
P = 2  # don't change this

X1 = np.random.standard_normal(size=[N, D])
X2 = np.random.standard_normal(size=[N, D])
idx_col = np.array([False, True] * 5).reshape([10, 1])  # bond to P=2

X1_flat = np.hstack((X1, idx_col))
X2_flat = np.hstack((X2, idx_col))


def test_coregionalization1Lkernel():
    kernel = Coregionalization1LKernel(
        base_variance=1.0,
        base_lengthscale=1.0,
        W_rank=2,
        input_dimension=D,
        output_dimension=P,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="test",
    )
    k_mat52 = gpflow.kernels.Matern52(1.0, 1.0)
    k_cor = gpflow.kernels.Coregion(output_dim=P, rank=2)
    assert np.allclose(kernel(X1_flat, X2_flat), k_mat52(X1, X2) * k_cor(idx_col, idx_col))
    assert np.allclose(kernel(X1_flat, full_cov=False), k_mat52(X1, full_cov=False) * k_cor(idx_col, full_cov=False))


def test_coregionalizationSOkernel():
    kernel = CoregionalizationSOKernel(
        [1.0, 0.9],  # bond to D=2
        [0.8, 0.7],  # bond to D=2
        D,
        P,
        add_error_kernel=True,
        add_prior=True,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="hello",
    )
    W = np.hstack([kernel.kernel.W_list[i].numpy().reshape([P, -1]) for i in range(D)])  # [P, D]
    error_var = kernel.kernel.error_variance.numpy()

    k_mat52_L1 = gpflow.kernels.Matern52(1.0, 0.8)
    k_mat52_L2 = gpflow.kernels.Matern52(0.9, 0.7)
    assert np.allclose(
        kernel(X1_flat, X2_flat),
        W[idx_col.astype(int), 0].T * k_mat52_L1(X1, X2) * W[idx_col.astype(int), 0]
        + W[idx_col.astype(int), 1].T * k_mat52_L2(X1, X2) * W[idx_col.astype(int), 1]
        + error_var * np.outer(idx_col.astype(float), idx_col.astype(float)),
    )
    assert np.allclose(
        kernel(X1_flat, full_cov=False),
        1.0 * W[idx_col.astype(int), 0].reshape(-1) ** 2
        + 0.9 * W[idx_col.astype(int), 1].reshape(-1) ** 2
        + error_var * idx_col.astype(float).reshape(-1),
    )


def test_coregionalizationMOkernel():
    kernel = CoregionalizationMOKernel(
        [1.0, 0.9],  # bond to D=2
        [0.8, 0.7],  # bond to D=2
        D,
        P,
        add_error_kernel=True,
        add_prior=True,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="hello",
    )
    W = np.hstack([kernel.kernel.W_list[i].numpy().reshape([P, -1]) for i in range(D)])  # [P, D]

    k_gpflow = gpflow.kernels.multioutput.LinearCoregionalization(
        [gpflow.kernels.Matern52(1.0, 0.8), gpflow.kernels.Matern52(0.9, 0.7)], W
    )
    assert np.allclose(
        kernel(X1, X2, full_cov=True, full_output_cov=True), k_gpflow(X1, X2, full_cov=True, full_output_cov=True)
    )
    assert np.allclose(
        kernel(X1, full_cov=False, full_output_cov=True), k_gpflow(X1, full_cov=False, full_output_cov=True)
    )


def test_coregionalizationPLkernel():
    kernel = CoregionalizationPLKernel(
        base_variance=1.0,
        base_lengthscale=1.0,
        W_rank=2,
        input_dimension=D,
        output_dimension=P,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="test",
    )
    k_mat52 = gpflow.kernels.Matern52(1.0, 1.0)
    k_cor = gpflow.kernels.Coregion(output_dim=P, rank=2)
    assert np.allclose(kernel(X1_flat, X2_flat), P * k_mat52(X1, X2) * k_cor(idx_col, idx_col))
    assert np.allclose(
        kernel(X1_flat, full_cov=False), P * k_mat52(X1, full_cov=False) * k_cor(idx_col, full_cov=False)
    )


def test_flexible_transfer_kernel():
    kernel = FlexibleTransferKernel(
        variance_list=[1.0, 0.1, 0.9],
        lengthscale_list=[0.8, 0.4, 0.7],
        input_dimension=D,
        output_dimension=P,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="test",
    )
    k_source = gpflow.kernels.Matern52(1.0, 0.8)
    k_st = gpflow.kernels.Matern52(0.1, 0.4)
    k_target = gpflow.kernels.Matern52(0.9, 0.7)

    gram_result = np.zeros([N, N], dtype=float)

    gram_source_x = np.zeros([N - np.sum(idx_col), N], dtype=float)
    gram_source_x[:, ~idx_col.reshape(-1)] = k_source(X1[~idx_col.reshape(-1)], X2[~idx_col.reshape(-1)])
    gram_source_x[:, idx_col.reshape(-1)] = k_st(X1[~idx_col.reshape(-1)], X2[idx_col.reshape(-1)])
    gram_result[~idx_col.reshape(-1), :] = gram_source_x

    gram_target_x = np.zeros([np.sum(idx_col), N], dtype=float)
    gram_target_x[:, ~idx_col.reshape(-1)] = k_st(X1[idx_col.reshape(-1)], X2[~idx_col.reshape(-1)])
    gram_target_x[:, idx_col.reshape(-1)] = k_target(X1[idx_col.reshape(-1)], X2[idx_col.reshape(-1)])
    gram_result[idx_col.reshape(-1), :] = gram_target_x

    assert np.allclose(kernel(X1_flat, X2_flat), gram_result)

    gram_result_diag = np.zeros(N, dtype=float)

    gram_result_diag[~idx_col.reshape(-1)] = k_source(X1[~idx_col.reshape(-1)], full_cov=False)
    gram_result_diag[idx_col.reshape(-1)] = k_target(X1[idx_col.reshape(-1)], full_cov=False)

    assert np.allclose(kernel(X1_flat, full_cov=False), gram_result_diag)


def test_miakernel():
    kernel = MIAdditiveKernel(
        variance_list=[1.0, 0.9],
        lengthscale_list=[0.8, 0.7],
        input_dimension=D,
        output_dimension=P,
        add_prior=False,
        lengthscale_prior_parameters=(1, 9),
        variance_prior_parameters=(1, 0.3),
        latent_kernel=LatentKernel.MATERN52,
        active_on_single_dimension=False,
        active_dimension=None,
        name="test",
    )
    k_source = gpflow.kernels.Matern52(1.0, 0.8)
    output_filter = gpflow.kernels.Linear()
    k_difference = gpflow.kernels.Matern52(0.9, 0.7)

    assert np.allclose(
        kernel(X1_flat, X2_flat),
        k_source(X1, X2) + output_filter(idx_col.astype(float), idx_col.astype(float)) * k_difference(X1, X2),
    )
    assert np.allclose(
        kernel(X1_flat, full_cov=False),
        k_source(X1, full_cov=False)
        + output_filter(idx_col.astype(float), full_cov=False) * k_difference(X1, full_cov=False),
    )


if __name__ == "__main__":
    test_coregionalization1Lkernel()
    test_coregionalizationSOkernel()
    test_coregionalizationMOkernel()
    test_coregionalizationPLkernel()
    test_flexible_transfer_kernel()
    test_miakernel()
