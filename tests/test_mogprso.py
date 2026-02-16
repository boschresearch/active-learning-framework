import numpy as np
import gpflow
from alef.kernels.multi_output_kernels.coregionalization_kernel import CoregionalizationSOKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from alef.models.mo_gpr_so import SOMOGPR

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
# mean function
m = gpflow.mean_functions.Constant()
# model
model = SOMOGPR((Xa, Ya), kernel, mean_function=m, noise_variance=[0.2, 0.3])
W_value = np.eye(P)
for l in range(P):
    model.kernel.kernel.W_list[l].assign(W_value[:, l])

# comparison model
k1 = gpflow.kernels.Matern52(1.3, 0.9)
k2 = gpflow.kernels.Matern52(0.8, 0.75)
mm1 = gpflow.models.GPR((Xa[0::2, :3], Ya[0::2, 0, None]), k1, mean_function=m, noise_variance=0.2)
mm2 = gpflow.models.GPR((Xa[1::2, :3], Ya[1::2, 0, None]), k2, mean_function=m, noise_variance=0.3)


def test_log_marginal_likelihood():
    MO_log_marg_lik = model.log_marginal_likelihood().numpy()
    SO1_log_marg_lik = mm1.log_marginal_likelihood().numpy()
    SO2_log_marg_lik = mm2.log_marginal_likelihood().numpy()

    assert np.allclose(MO_log_marg_lik, SO1_log_marg_lik + SO2_log_marg_lik)


def test_predict_f():
    Xt0 = np.random.normal(size=[10, D])
    Xt1 = np.random.normal(size=[10, D])

    mu, cov = model.predict_f(
        np.vstack((np.hstack((Xt0, np.zeros([10, 1]))), np.hstack((Xt1, np.ones([10, 1]))))), full_cov=True
    )

    mu1, cov1 = mm1.predict_f(Xt0, full_cov=True)
    mu2, cov2 = mm2.predict_f(Xt1, full_cov=True)

    assert np.allclose(mu[:10], mu1)
    assert np.allclose(mu[10:], mu2)
    assert np.allclose(cov[0, :10, :10], cov1[0, ...])
    assert np.allclose(cov[0, 10:, 10:], cov2[0, ...])

    mu, var = model.predict_f(
        np.vstack((np.hstack((Xt0, np.zeros([10, 1]))), np.hstack((Xt1, np.ones([10, 1]))))), full_cov=False
    )
    mu1, var1 = mm1.predict_f(Xt0, full_cov=False)
    mu2, var2 = mm2.predict_f(Xt1, full_cov=False)

    assert np.allclose(mu[:10], mu1)
    assert np.allclose(mu[10:], mu2)
    assert np.allclose(var[:10, 0], var1[..., 0])
    assert np.allclose(var[10:, 0], var2[..., 0])


def test_predict_y():
    Xt0 = np.random.normal(size=[10, D])
    Xt1 = np.random.normal(size=[10, D])

    mu, var = model.predict_y(
        np.vstack((np.hstack((Xt0, np.zeros([10, 1]))), np.hstack((Xt1, np.ones([10, 1]))))), full_cov=False
    )
    mu1, var1 = mm1.predict_y(Xt0, full_cov=False)
    mu2, var2 = mm2.predict_y(Xt1, full_cov=False)

    assert np.allclose(mu[:10], mu1)
    assert np.allclose(mu[10:], mu2)
    assert np.allclose(var[:10, 0], var1[..., 0])
    assert np.allclose(var[10:, 0], var2[..., 0])


if __name__ == "__main__":
    test_log_marginal_likelihood()
    test_predict_f()
    test_predict_y()
