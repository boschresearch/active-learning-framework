import numpy as np
import torch
import pytest
from pyro.distributions import MultivariateNormal
from alef.utils.pyro_distributions import MultivariateNormalSVD

N_batch = 20
expand_batch = [30, 20]
D_prior = 10
D_post = 5
D = D_prior + D_post

L = np.random.uniform(0, 1, [D, D])
cov = L @ L.T + np.eye(D)
cov = torch.from_numpy(cov).to(torch.get_default_dtype())
loc = np.random.standard_normal([N_batch, D])
loc = torch.from_numpy(loc).to(cov.dtype)

test_values = torch.from_numpy(np.random.standard_normal([N_batch, D_post])).to(torch.get_default_dtype())
test_values_after_expand = torch.from_numpy(np.random.standard_normal(expand_batch + [D_post])).to(
    torch.get_default_dtype()
)


tmp = MultivariateNormal(loc, covariance_matrix=cov)
Y = tmp.sample()
L_prior = torch.linalg.cholesky(cov[..., :D_prior, :D_prior])
u, s, vh = torch.linalg.svd(cov[..., :D_prior, :D_prior])
K_prior_new = cov[..., :D_prior, D_prior:]
K_new = cov[..., D_prior:, D_prior:]


@pytest.mark.parametrize(
    "dist1,dist2",
    [
        (
            MultivariateNormal(loc[..., D_prior:], covariance_matrix=K_new),
            MultivariateNormalSVD(loc[..., D_prior:], covariance_matrix=K_new),
        ),
        # (
        #    GPMatheron(loc[..., D_prior:], loc[..., :D_prior], Y[..., :D_prior], L_prior, K_prior_new, K_new),
        #    GPMatheronSVD(loc[..., D_prior:], loc[..., :D_prior], Y[..., :D_prior], u, torch.sqrt(s), K_prior_new, K_new)
        # ),
    ],
)
def test_equivalent_distribution(dist1, dist2):
    def run_assert(dist1, dist2, eval_tensor):
        assert dist1.sample().size() == dist2.sample().size()
        assert (
            dist1.sample(
                [
                    2,
                    3,
                ]
            ).size()
            == dist2.sample(
                [
                    2,
                    3,
                ]
            ).size()
        )
        assert torch.allclose(dist1.mean, dist2.mean, rtol=1e-4)
        assert torch.allclose(dist1.covariance_matrix, dist2.covariance_matrix, rtol=1e-4)
        assert torch.allclose(dist1.variance, dist2.variance, rtol=1e-4)
        assert torch.allclose(dist1.entropy(), dist2.entropy(), rtol=1e-4)
        assert torch.allclose(dist1.log_prob(eval_tensor), dist2.log_prob(eval_tensor), rtol=1e-4)

    run_assert(dist1, dist2, test_values)
    run_assert(dist1.expand(expand_batch), dist2.expand(expand_batch), test_values_after_expand)


if __name__ == "__main__":
    test_equivalent_distribution(
        MultivariateNormal(loc[..., D_prior:], covariance_matrix=cov[..., D_prior:, D_prior:]),
        MultivariateNormalSVD(loc[..., D_prior:], covariance_matrix=cov[..., D_prior:, D_prior:]),
    )

    # test_equivalent_distribution(
    #    GPMatheron(loc[..., D_prior:], loc[..., :D_prior], Y[..., :D_prior], L_prior, K_prior_new, K_new),
    #    GPMatheronSVD(loc[..., D_prior:], loc[..., :D_prior], Y[..., :D_prior], u, torch.sqrt(s), K_prior_new, K_new)
    # )
