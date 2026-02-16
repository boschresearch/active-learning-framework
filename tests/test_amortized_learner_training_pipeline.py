import numpy as np
import pytest
import torch
import gpytorch
from pyro import poutine, condition
from pyro.distributions import MultivariateNormal
from alef.models.gp_model_pytorch import ExactGPModel
from alef.configs.active_learner.amortized_policies.loss_configs import (
    ContinuousDADLossConfig,
    ContinuousScoreDADLossConfig,
    GPEntropyLoss1Config,
    GPEntropyLoss2Config,
    GPMILoss1Config,
    GPMILoss2Config,
    GPMI_EntropyLoss1Config,
)
from alef.configs.active_learner.amortized_policies.policy_configs import (
    ContinuousGPPolicyConfig,
)

# need GP model
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType

from alef.active_learner.amortized_policies.amortized_policy_factory import AmortizedPolicyFactory
from alef.active_learner.amortized_policies.simulated_processes.multiple_steps.continuous_gp_al import (
    PytestSequentialGaussianProcessContinuousDomain,
)
from alef.active_learner.amortized_policies.loss.multiple_steps.oed_mi import PriorContrastiveEstimation
from alef.active_learner.amortized_policies.loss.multiple_steps.gp_entropy import (
    GPEntroopy1,
    GPEntroopy2,
)
from alef.active_learner.amortized_policies.loss.multiple_steps.gp_mi import (
    GPMutualInformation1,
    GPMutualInformation2,
)
from alef.active_learner.amortized_policies.training.main_train import Trainer

config_args = {"batch_size": 3, "num_kernels": 4, "num_functions_per_kernel": 5, "num_epochs": 6, "epochs_size": 7}

T = 2
D = 2
device = "cpu"
assert device == "cpu", "do not support gpu test at the moment, pyro site control will be wrong (only wrong in test)"
## test loss functions
design_net = AmortizedPolicyFactory.build(
    ContinuousGPPolicyConfig(
        input_dim=D,
        hidden_dim_encoder=32,
        encoding_dim=8,
        hidden_dim_emitter=32,
        self_attention_layer=True,
        domain_warpper=DomainWarpperType.TANH,
        device=device,
    )
)
process = PytestSequentialGaussianProcessContinuousDomain(
    design_net,
    kernel_config=RBFWithPriorPytorchConfig(input_dimension=D, base_lengthscale=[0.6, 0.2]),
    n_steps=T,
    sample_gp_prior=True,
    device=device,
)

Nk, Nf, B = config_args["num_kernels"], config_args["num_functions_per_kernel"], config_args["batch_size"]
X = torch.rand([B, Nk, Nf, T, D])
Y = torch.randn([B, Nk, Nf, T])
X_grid = torch.zeros([B, Nk, Nf, 2, D])
X_grid[:, :, :, 1, :] = 1
Y_grid = torch.ones([B, Nk, Nf, 2])


def test_amortized_al_dad_loss():
    loss = PriorContrastiveEstimation(
        B,
        Nk,
        Nf,
        data_source=iter(
            [
                {
                    **{f"Default_xi{i + 1}": X[:, 0, None, 0, None, i, None, :] for i in range(T)},
                    **{f"Default_y{i + 1}": Y[:, 0, None, 0, None, i, None] for i in range(T)},
                },
                {
                    **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                    **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
                },
            ]
        ),
    )
    loss.differentiable_loss(process)  # this method execute process twice
    # 1st run get primary p(Y) with shape [B, 1, 1], the kernel is not available (which is why we don't test the value)
    # 2nd run get contrastive p(Y) with shape [B, Nk, Nf], the kernels are available

    # then test cross validation
    rmse_mean, rmse_stderr = loss.validation(process)
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance

    # compute ref value
    rmse_ref = []
    for i, k in enumerate(kernel_list):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_var[i]
        model = ExactGPModel(X[:, i, ...], Y[:, i, ...], likelihood, gpytorch.means.ZeroMean(), k)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mu_pred = model(X_grid[:, i, ...]).mean
            rmse = (mu_pred - Y_grid[:, i]).pow(2).mean(-1).sqrt()
            rmse_ref.append(rmse.flatten())
    with torch.no_grad():
        rmse_ref = torch.cat(rmse_ref)
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


def test_amortized_al_entropy_loss1():
    loss = GPEntroopy1(B, Nk, Nf)
    trace = poutine.trace(
        condition(
            loss.differentiable_loss,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    loss_value = trace.nodes["_RETURN"]["value"]

    # compute reference value
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance
    K = torch.cat(
        [
            kernel(X[:, None, i, ...]).to_dense() + noise_var[i] * torch.eye(X.shape[-2], device=X.device)
            for i, kernel in enumerate(kernel_list)
        ],
        dim=1,
    )  # [B, Nk, Nf, T, T]
    assert torch.isclose(
        -loss_value, MultivariateNormal(torch.zeros_like(Y), covariance_matrix=K).entropy().mean(), rtol=1e-4, atol=1e-5
    )

    # then test cross validation
    trace = poutine.trace(
        condition(
            loss.validation,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes["_RETURN"]["value"]
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance

    # compute ref value
    rmse_ref = []
    for i, k in enumerate(kernel_list):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_var[i]
        model = ExactGPModel(X[:, i, ...], Y[:, i, ...], likelihood, gpytorch.means.ZeroMean(), k)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mu_pred = model(X_grid[:, i, ...]).mean
            rmse = (mu_pred - Y_grid[:, i]).pow(2).mean(-1).sqrt()
            rmse_ref.append(rmse.flatten())
    with torch.no_grad():
        rmse_ref = torch.cat(rmse_ref)
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


def test_amortized_al_entropy_loss2():
    loss = GPEntroopy2(B, Nk, Nf)
    trace = poutine.trace(
        condition(
            loss.differentiable_loss,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    loss_value = trace.nodes["_RETURN"]["value"]

    # compute reference value
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance
    K = torch.cat(
        [
            kernel(X[:, None, i, ...]).to_dense() + noise_var[i] * torch.eye(X.shape[-2], device=X.device)
            for i, kernel in enumerate(kernel_list)
        ],
        dim=1,
    )  # [B, Nk, Nf, T, T]

    log_pY = MultivariateNormal(torch.zeros_like(Y), covariance_matrix=K).log_prob(Y).mean()

    assert torch.isclose(loss_value, log_pY, rtol=1e-4, atol=1e-5)

    # then test cross validation
    trace = poutine.trace(
        condition(
            loss.validation,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes["_RETURN"]["value"]
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance

    # compute ref value
    rmse_ref = []
    for i, k in enumerate(kernel_list):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_var[i]
        model = ExactGPModel(X[:, i, ...], Y[:, i, ...], likelihood, gpytorch.means.ZeroMean(), k)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mu_pred = model(X_grid[:, i, ...]).mean
            rmse = (mu_pred - Y_grid[:, i]).pow(2).mean(-1).sqrt()
            rmse_ref.append(rmse.flatten())
    with torch.no_grad():
        rmse_ref = torch.cat(rmse_ref)
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


def test_amortized_al_gpmi_loss1():
    loss = GPMutualInformation1(B, Nk, Nf)
    trace = poutine.trace(
        condition(
            loss.differentiable_loss,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    loss_value = trace.nodes["_RETURN"]["value"]

    # compute reference value
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance
    K = torch.cat(
        [
            kernel(X[:, None, i, ...]).to_dense() + noise_var[i] * torch.eye(X.shape[-2], device=X.device)
            for i, kernel in enumerate(kernel_list)
        ],
        dim=1,
    )  # [B, Nk, Nf, T, T]

    K_cross = torch.cat(
        [
            kernel(X_grid[:, None, i, ...], X[:, None, i, ...]).to_dense()  # [B, Nf, n_grid, T]
            for i, kernel in enumerate(kernel_list)
        ],
        dim=1,
    )  # [B, Nk, Nf, n_grid, T]
    K_grid = torch.cat(
        [
            kernel(X_grid[:, None, i, ...]).to_dense()
            + noise_var[i] * torch.eye(X_grid.shape[-2], device=X_grid.device)  # [B, Nf, n_grid, n_grid]
            for i, kernel in enumerate(kernel_list)
        ],
        dim=1,
    )  # [B, Nk, Nf, n_grid, n_grid]
    cholesky = torch.linalg.cholesky(K_grid)
    K_right = torch.linalg.solve_triangular(cholesky, K_cross, upper=False)  # [B, Nk, Nf, n_grid, T]

    mu_y_given_y_grid = torch.matmul(
        K_right.transpose(-1, -2), torch.linalg.solve_triangular(cholesky, Y_grid.unsqueeze(-1), upper=False)
    ).squeeze(-1)
    Cov_y_given_y_grid = K - torch.matmul(K_right.transpose(-1, -2), K_right)  # [B, Nk, Nf, T, T]

    entropy = MultivariateNormal(torch.zeros_like(Y), covariance_matrix=K).entropy().mean()
    entropy_y_given_y_grid = (
        MultivariateNormal(mu_y_given_y_grid, covariance_matrix=Cov_y_given_y_grid).entropy().mean()
    )
    assert torch.isclose(-loss_value, entropy - entropy_y_given_y_grid, rtol=1e-4, atol=1e-5)

    # then test cross validation
    trace = poutine.trace(
        condition(
            loss.validation,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes["_RETURN"]["value"]
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance

    # compute ref value
    rmse_ref = []
    for i, k in enumerate(kernel_list):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_var[i]
        model = ExactGPModel(X[:, i, ...], Y[:, i, ...], likelihood, gpytorch.means.ZeroMean(), k)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mu_pred = model(X_grid[:, i, ...]).mean
            rmse = (mu_pred - Y_grid[:, i]).pow(2).mean(-1).sqrt()
            rmse_ref.append(rmse.flatten())
    with torch.no_grad():
        rmse_ref = torch.cat(rmse_ref)
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


def test_amortized_al_gpmi_loss2():
    loss = GPMutualInformation2(B, Nk, Nf)
    trace = poutine.trace(
        condition(
            loss.differentiable_loss,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    loss_value = trace.nodes["_RETURN"]["value"]

    # compute reference value
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance
    K = torch.cat(
        [
            kernel(X[:, None, i, ...]).to_dense() + noise_var[i] * torch.eye(X.shape[-2], device=X.device)
            for i, kernel in enumerate(kernel_list)
        ],
        dim=1,
    )  # [B, Nk, Nf, T, T]

    K_cross = torch.cat(
        [kernel(X_grid[:, None, i, ...], X[:, None, i, ...]).to_dense() for i, kernel in enumerate(kernel_list)], dim=1
    )  # [B, Nk, Nf, n_grid, T]
    K_grid = torch.cat(
        [
            kernel(X_grid[:, None, i, ...]).to_dense()
            + noise_var[i] * torch.eye(X_grid.shape[-2], device=X_grid.device)
            for i, kernel in enumerate(kernel_list)
        ],
        dim=1,
    )  # [B, Nk, Nf, n_grid, n_grid]
    cholesky = torch.linalg.cholesky(K_grid)
    K_right = torch.linalg.solve_triangular(cholesky, K_cross, upper=False)  # [B, Nk, Nf, n_grid, T]

    mu_y_given_y_grid = torch.matmul(
        K_right.transpose(-1, -2), torch.linalg.solve_triangular(cholesky, Y_grid.unsqueeze(-1), upper=False)
    ).squeeze(-1)
    Cov_y_given_y_grid = K - torch.matmul(K_right.transpose(-1, -2), K_right)  # [B, Nk, Nf, T, T]

    log_pY = MultivariateNormal(torch.zeros_like(Y), covariance_matrix=K).log_prob(Y).mean()
    log_pY_given_y_grid = MultivariateNormal(mu_y_given_y_grid, covariance_matrix=Cov_y_given_y_grid).log_prob(Y).mean()
    assert torch.isclose(loss_value, log_pY - log_pY_given_y_grid, rtol=1e-4, atol=1e-5)

    # then test cross validation
    trace = poutine.trace(
        condition(
            loss.validation,
            data={
                **{f"Default_xi{i + 1}": X[:, :, :, i, None, :] for i in range(T)},
                **{f"Default_y{i + 1}": Y[:, :, :, i, None] for i in range(T)},
            },
        ),
        graph_type="flat",
    ).get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes["_RETURN"]["value"]
    kernel_list = process.gp_dist.kernel_list
    noise_var = process.gp_dist.noise_variance

    # compute ref value
    rmse_ref = []
    for i, k in enumerate(kernel_list):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_var[i]
        model = ExactGPModel(X[:, i, ...], Y[:, i, ...], likelihood, gpytorch.means.ZeroMean(), k)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mu_pred = model(X_grid[:, i, ...]).mean
            rmse = (mu_pred - Y_grid[:, i]).pow(2).mean(-1).sqrt()
            rmse_ref.append(rmse.flatten())
    with torch.no_grad():
        rmse_ref = torch.cat(rmse_ref)
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


@pytest.mark.parametrize(
    "loss_config",
    (
        ContinuousDADLossConfig(**config_args),
        ContinuousScoreDADLossConfig(**config_args),
        GPEntropyLoss1Config(**config_args),
        GPEntropyLoss2Config(**config_args),
        GPMILoss1Config(**config_args),
        GPMILoss2Config(**config_args),
        GPMI_EntropyLoss1Config(loss_config_list=[GPMILoss1Config(**config_args), GPEntropyLoss1Config(**config_args)]),
    ),
)
def test_amortized_al_main_trainer(tmp_path, loss_config):
    D = 2
    kernel_config = RBFWithPriorPytorchConfig(input_dimension=D, base_lengthscale=[0.6, 0.2])

    trainer = Trainer(log_path=tmp_path, seed=0, device=device, fast_tqdm=True)

    trainer.set_policy_config(input_dimension=D, self_attention_layer=True, domain_warpper=DomainWarpperType.TANH)
    trainer.set_loss_config(loss_config.__class__.__name__)  # just to test if this method operates
    assert isinstance(trainer.loss_config, loss_config.__class__)
    trainer.loss_config = loss_config  # we assign smaller batch for pytest
    trainer.set_training_config(
        "AmortizedNonmyopicContinuousRandomGPPolicyTrainingConfig",
        kernel_config,
        20,  # number of queries during training
    )
    trainer.save_settings()
    trainer.train()  # train and save


if __name__ == "__main__":
    test_amortized_al_dad_loss()
    test_amortized_al_entropy_loss1()
    test_amortized_al_entropy_loss2()
    test_amortized_al_gpmi_loss1()
    test_amortized_al_gpmi_loss2()
