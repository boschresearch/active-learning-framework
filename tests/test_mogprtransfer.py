import numpy as np
import tensorflow as tf
import gpflow
from alef.kernels.multi_output_kernels.coregionalization_kernel import CoregionalizationSOKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from alef.models.mo_gpr_so import SOMOGPR
from alef.models.mo_gpr_transfer import TransferGPR

N = 50
D = 3
P = 2

# data
X = np.random.normal(size=[N * P, D])
Y = np.random.normal(size=[N * P, 1]) * 3
p = np.array([False, True] * N)
p = p.reshape([-1, 1])
Xa = np.hstack((X, p))
Ya = np.hstack((Y, p))

# kernel
kernel = CoregionalizationSOKernel(
    [1.3, 0.8],
    [0.9, 0.75],
    D,
    P,
    latent_kernel=LatentKernel.MATERN52,
    add_error_kernel=False,
    add_prior=True,
    lengthscale_prior_parameters=(1, 9),
    variance_prior_parameters=(1, 0.3),
    active_on_single_dimension=False,
    active_dimension=None,
    name=None,
)
for l in range(P):
    kernel.kernel.W_list[l].assign(np.random.standard_normal(size=[P]))

# mean function
m = gpflow.mean_functions.Constant()
# model
model = TransferGPR(
    (Xa[Xa[:, -1] == 0], Ya[Xa[:, -1] == 0]),
    (Xa[Xa[:, -1] == 1], Ya[Xa[:, -1] == 1]),
    kernel,
    mean_function=m,
    noise_variance=[0.2, 0.3],
)
model_ref = SOMOGPR((Xa, Ya), kernel, mean_function=m, noise_variance=[0.2, 0.3])


def test_raw_log_marginal_likelihood():
    log_marg_lik = model.log_marginal_likelihood().numpy()
    log_marg_lik_ref = model_ref.log_marginal_likelihood().numpy()

    assert np.allclose(log_marg_lik, log_marg_lik_ref)


def test_predict_f():
    Xt0 = np.random.normal(size=[10, D])
    Xt1 = np.random.normal(size=[10, D])

    mu, cov = model.predict_f(
        np.vstack((np.hstack((Xt0, np.zeros([10, 1]))), np.hstack((Xt1, np.ones([10, 1]))))), full_cov=True
    )

    mu_ref, cov_ref = model_ref.predict_f(
        np.vstack((np.hstack((Xt0, np.zeros([10, 1]))), np.hstack((Xt1, np.ones([10, 1]))))), full_cov=True
    )

    assert np.allclose(mu, mu_ref)
    assert np.allclose(cov, cov_ref)


def test_predict_y():
    Xt0 = np.random.normal(size=[10, D])
    Xt1 = np.random.normal(size=[10, D])

    mu, cov = model.predict_y(
        np.vstack((np.hstack((Xt0, np.zeros([10, 1]))), np.hstack((Xt1, np.ones([10, 1]))))), full_cov=False
    )

    mu_ref, cov_ref = model_ref.predict_y(
        np.vstack((np.hstack((Xt0, np.zeros([10, 1]))), np.hstack((Xt1, np.ones([10, 1]))))), full_cov=False
    )

    assert np.allclose(mu, mu_ref)
    assert np.allclose(cov, cov_ref)


def test_source_cholesky():
    K = kernel(Xa[Xa[:, -1] == 0])
    L_ref = tf.linalg.cholesky(K + tf.linalg.diag(np.array([0.2] * N)))
    L = model.compute_source_cholesky()
    assert np.allclose(L, L_ref)


def test_full_cholesky():
    Ls = model.compute_source_cholesky()
    model.set_source_cholesky(Ls)
    K = kernel(np.vstack((Xa[Xa[:, -1] == 0], Xa[Xa[:, -1] == 1])))

    L_ref = tf.linalg.cholesky(K + tf.linalg.diag(np.array([0.2] * N + [0.3] * N)))
    L = model.full_gram_noisy_cholesky(Xa[Xa[:, -1] == 0], Xa[Xa[:, -1] == 1], Ls)
    assert np.allclose(L, L_ref)


def test_log_marginal_likelihood():
    model.set_source_cholesky(model.compute_source_cholesky())
    log_marg_lik = model.log_marginal_likelihood().numpy()
    log_marg_lik_ref = model_ref.log_marginal_likelihood().numpy()

    assert np.allclose(log_marg_lik, log_marg_lik_ref)


def test_after_precomputation():
    model.set_source_cholesky(model.compute_source_cholesky())
    test_predict_f()
    test_predict_y()


if __name__ == "__main__":
    test_raw_log_marginal_likelihood()
    test_predict_f()
    test_predict_y()
    test_source_cholesky()
    test_full_cholesky()
    test_log_marginal_likelihood()
    test_after_precomputation()
