# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
from numbers import Real
import torch
from torch.distributions import constraints, Distribution, ExponentialFamily
from torch.distributions.utils import _standard_normal, lazy_property
from pyro.distributions.torch_distribution import TorchDistributionMixin

"""
Part of the following code is adapted from
https://pytorch.org/docs/stable/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
"""


class MultivariateNormalSVD(Distribution, TorchDistributionMixin):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covariance_matrix": constraints.positive_semidefinite,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, covariance_matrix, validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")
        if covariance_matrix.dim() < 2:
            raise ValueError(
                "covariance_matrix must be at least two-dimensional, with optional leading batch dimensions"
            )
        batch_shape = torch.broadcast_shapes(covariance_matrix.shape[:-2], loc.shape[:-1])
        self.loc = loc.expand(batch_shape + (-1,))
        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        u, s, vh = torch.linalg.svd(covariance_matrix)
        sqrt_s = torch.sqrt(s)
        self._unbroadcasted_scale_tril = torch.matmul(u, torch.diag_embed(sqrt_s))
        self._u = u.expand(self._batch_shape + (-1, -1))
        self._sqrt_s = torch.sqrt(s).expand(self._batch_shape + (-1,))
        # self._vh = vh.expand(self._batch_shape + (-1, -1))

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateNormalSVD, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape

        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        new._u = self._u.expand(cov_shape)
        new._sqrt_s = self._sqrt_s.expand(loc_shape)
        # new._vh = self._vh.expand(cov_shape)
        super(MultivariateNormalSVD, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def precision_matrix(self):
        # u.T = u inverse
        # (u sqrt_s)^-1 = (1/sqrt_s) u.T
        return torch.matmul(torch.diag_embed(torch.reciprocal(self._sqrt_s)), self._u.mT)

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + torch.matmul(self._unbroadcasted_scale_tril, eps.unsqueeze(-1)).squeeze(-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M_sqrt = torch.matmul(self.precision_matrix, diff.unsqueeze(-1)).squeeze(-1)
        M = torch.pow(M_sqrt, 2.0).sum(-1)
        half_log_det = self._sqrt_s.log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = self._sqrt_s.log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)


class GPMatheron(Distribution, TorchDistributionMixin):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution,
    where we sample posterior points given already sampled prior points.

    Args:
        loc_new (Tensor): mean of distribution at sampling locations
        loc_prior (Tensor): mean of distribution used to sample the prior points
        y_prior (Tensor): prior samples output points
        covariance_prior_new (Tensor): k(x_prior, x_new)
        scale_tril_prior (Tensor): cholesky( k(x_prior, x_prior) + var_prior * I )
        covariance_new (Tensor): k(x_post, x_new) + var_post * I
    """

    arg_constraints = {
        "loc": constraints.real_vector,
        "loc_new": constraints.real_vector,
        "loc_prior": constraints.real_vector,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc_new,
        loc_prior,
        y_prior,
        scale_tril_prior,  # k(x_prior, x_prior) + var_prior * I
        covariance_prior_new,  # k(x_prior, x_posterior)
        covariance_new,  # k(x_post, x_post) + var_post * I
        validate_args=None,
    ):
        batch_shape = torch.broadcast_shapes(
            loc_new.shape[:-1],
            loc_prior.shape[:-1],
            y_prior.shape[:-1],
            scale_tril_prior.shape[:-2],
            covariance_prior_new.shape[:-2],
            covariance_new.shape[:-2],
        )

        self.loc_prior = loc_prior.expand(batch_shape + (-1,))
        self.loc_new = loc_new.expand(batch_shape + (-1,))
        self.y_prior = y_prior.expand(batch_shape + (-1,))

        event_shape = self.loc_new.shape[-1:]

        self._Lxx = scale_tril_prior
        # self._Kxt = covariance_prior_new
        # self._Ktt = covariance_new

        self._L_inv_Kxt = torch.linalg.solve_triangular(scale_tril_prior, covariance_prior_new, upper=False)
        Ktx_L_T_inv = torch.einsum("...ij->...ji", self._L_inv_Kxt)
        cor = covariance_new - torch.matmul(Ktx_L_T_inv, self._L_inv_Kxt)
        L_post_cor, info = torch.linalg.cholesky_ex(cor)
        self._L_full = torch.cat(
            [
                torch.cat((scale_tril_prior, torch.zeros_like(self._L_inv_Kxt)), dim=-1),
                torch.cat((Ktx_L_T_inv, L_post_cor), dim=-1),
            ],
            dim=-2,
        )

        epsi = y_prior - loc_prior
        L_inv_epsi = torch.linalg.solve_triangular(scale_tril_prior, torch.unsqueeze(epsi, dim=-1), upper=False)
        self.loc = self.loc_new + torch.matmul(self._L_inv_Kxt.mT, L_inv_epsi).squeeze(dim=-1)
        self._unbroadcasted_scale_tril = L_post_cor
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GPMatheron, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_prior_shape = batch_shape + self.loc_prior.shape[-1:]
        loc_shape = batch_shape + self.event_shape
        batch_shape + self.event_shape + self.event_shape

        new.loc_prior = self.loc_prior.expand(loc_prior_shape)
        new.loc_new = self.loc_new.expand(loc_shape)
        new.y_prior = self.y_prior.expand(loc_prior_shape)
        new._Lxx = self._Lxx
        new._L_inv_Kxt = self._L_inv_Kxt
        new._L_full = self._L_full
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        super(GPMatheron, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def precision_matrix(self):
        return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return self._unbroadcasted_scale_tril.pow(2).sum(-1).expand(self._batch_shape + self._event_shape)

    @property
    def _prior_post_event_shape(self):
        return torch.cat([self.loc_prior, self.loc_new], dim=-1).shape[-1:]

    def rsample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        # first sample from standard normal
        # shape = [sample_shape, batch_shape, prior_length + post_length]
        shape = sample_shape + self._batch_shape + self._prior_post_event_shape
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # compute self._L_full @ eps
        # this is equivalent to sampling from N([loc_prior, loc_new], L_full @ L_full.mT)
        f_all = torch.cat([self.loc_prior, self.loc_new], dim=-1) + torch.matmul(
            self._L_full, eps.unsqueeze(-1)
        ).squeeze(-1)
        n = self.loc_prior.size(-1)
        f_new = f_all[..., n:]
        y = f_all[..., :n]
        # compute matheron  f_new | self.y_prior = f_new + Ktx @ (Lxx @ Lxx.T)^-1 @ (self.y_prior - y)
        return f_new + torch.matmul(
            self._L_inv_Kxt.mT,
            torch.linalg.solve_triangular(self._Lxx, self.y_prior.unsqueeze(-1) - y.unsqueeze(-1), upper=False),
        ).squeeze(-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M_sqrt = torch.linalg.solve_triangular(self.scale_tril, diff.unsqueeze(-1), upper=False).squeeze(-1)
        M = torch.pow(M_sqrt, 2.0).sum(-1)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)


class GPMatheronSVD(Distribution, TorchDistributionMixin):
    arg_constraints = {
        "loc": constraints.real_vector,
        "loc_new": constraints.real_vector,
        "loc_prior": constraints.real_vector,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(
        self,
        loc_new,
        loc_prior,
        y_prior,
        U_prior,  # U of k(x_prior, x_prior) + var_prior * I (SVD decomp)
        S_sqrt_prior,  # S_sqrt of k(x_prior, x_prior) + var_prior * I (SVD decomp)
        covariance_prior_new,  # k(x_prior, x_posterior)
        covariance_new,  # k(x_post, x_post) + var_post * I
        validate_args=None,
    ):
        batch_shape = torch.broadcast_shapes(
            loc_new.shape[:-1],
            loc_prior.shape[:-1],
            y_prior.shape[:-1],
            U_prior.shape[:-2],
            S_sqrt_prior.shape[:-1],
            covariance_prior_new.shape[:-2],
            covariance_new.shape[:-2],
        )

        self.loc_prior = loc_prior.expand(batch_shape + (-1,))
        self.loc_new = loc_new.expand(batch_shape + (-1,))
        self.y_prior = y_prior.expand(batch_shape + (-1,))

        self._prior_sqrt_s_inv_u_inv = torch.matmul(
            torch.diag_embed(torch.reciprocal(S_sqrt_prior)), torch.einsum("...ij->...ji", U_prior)
        )
        Us_inv_Kxt = torch.matmul(self._prior_sqrt_s_inv_u_inv, covariance_prior_new)
        self._Ktx_prior_u_sqrt_s_inv = torch.einsum("...ij->...ji", Us_inv_Kxt)
        cor = covariance_new - torch.matmul(self._Ktx_prior_u_sqrt_s_inv, Us_inv_Kxt)

        u, s, vh = torch.linalg.svd(cor)
        sqrt_s = torch.sqrt(s)
        self._unbroadcasted_scale_tril = torch.matmul(u, torch.diag_embed(sqrt_s))
        self._post_u = u.expand(batch_shape + (-1, -1))
        self._post_sqrt_s = torch.sqrt(s).expand(batch_shape + (-1,))

        self._L_full = torch.cat(
            [
                torch.cat(
                    (torch.matmul(U_prior, torch.diag_embed((S_sqrt_prior))), torch.zeros_like(Us_inv_Kxt)), dim=-1
                ),
                torch.cat((self._Ktx_prior_u_sqrt_s_inv, self._unbroadcasted_scale_tril), dim=-1),
            ],
            dim=-2,
        )

        epsi = y_prior - loc_prior
        L_inv_epsi = torch.matmul(self._prior_sqrt_s_inv_u_inv, epsi.unsqueeze(-1))
        self.loc = self.loc_new + torch.matmul(self._Ktx_prior_u_sqrt_s_inv, L_inv_epsi).squeeze(-1).expand(
            batch_shape + (-1,)
        )

        event_shape = self.loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GPMatheronSVD, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_prior_shape = batch_shape + self.loc_prior.shape[-1:]
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + self.event_shape

        new.loc_prior = self.loc_prior.expand(loc_prior_shape)
        new.loc_new = self.loc_new.expand(loc_shape)
        new.y_prior = self.y_prior.expand(loc_prior_shape)

        new._prior_sqrt_s_inv_u_inv = self._prior_sqrt_s_inv_u_inv
        new._Ktx_prior_u_sqrt_s_inv = self._Ktx_prior_u_sqrt_s_inv
        new._L_full = self._L_full

        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        new._post_u = self._post_u.expand(cov_shape)
        new._post_sqrt_s = self._post_sqrt_s.expand(loc_shape)
        # new._vh = self._vh.expand(cov_shape)
        super(GPMatheronSVD, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def covariance_matrix(self):
        return torch.matmul(self._unbroadcasted_scale_tril, self._unbroadcasted_scale_tril.mT).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def precision_matrix(self):
        return torch.matmul(torch.diag_embed(torch.reciprocal(self._post_sqrt_s)), self._post_u.mT)

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def variance(self):
        return torch.diagonal(self.covariance_matrix, dim1=-2, dim2=-1)

    @property
    def _prior_post_event_shape(self):
        return torch.cat([self.loc_prior, self.loc_new], dim=-1).shape[-1:]

    def rsample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        # first sample from standard normal
        # shape = [sample_shape, batch_shape, prior_length + post_length]
        shape = sample_shape + self._batch_shape + self._prior_post_event_shape
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # compute self._L_full @ eps
        # this is equivalent to sampling from N([loc_prior, loc_new], L_full @ L_full.mT)
        f_all = torch.cat([self.loc_prior, self.loc_new], dim=-1) + torch.matmul(
            self._L_full, eps.unsqueeze(-1)
        ).squeeze(-1)
        n = self.loc_prior.size(-1)
        f_new = f_all[..., n:]
        y = f_all[..., :n]
        # compute matheron  f_new | self.y_prior = f_new + Ktx @ (Lxx @ Lxx.T)^-1 @ (self.y_prior - y)
        return f_new + torch.matmul(
            self._Ktx_prior_u_sqrt_s_inv,
            torch.matmul(self._prior_sqrt_s_inv_u_inv, self.y_prior.unsqueeze(-1) - y.unsqueeze(-1)),
        ).squeeze(-1)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M_sqrt = torch.matmul(self.precision_matrix, diff.unsqueeze(-1)).squeeze(-1)
        M = torch.pow(M_sqrt, 2.0).sum(-1)
        half_log_det = self._post_sqrt_s.log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def entropy(self):
        half_log_det = self._post_sqrt_s.log().sum(-1)
        H = 0.5 * self._event_shape[0] * (1.0 + math.log(2 * math.pi)) + half_log_det
        if len(self._batch_shape) == 0:
            return H
        else:
            return H.expand(self._batch_shape)


############
### the following are uni-variate dists
class GPMatheronUnivariate(ExponentialFamily, TorchDistributionMixin):
    r"""
    Creates a multivariate normal (also called Gaussian) distribution,
    where we sample posterior points given already sampled prior points.

    Args:
        loc_new (float or Tensor): mean of distribution at sampling locations
        loc_prior (float or Tensor): mean of distribution used to sample the prior points
        y_prior (float or Tensor): prior samples output points
        covariance_prior_new (Tensor): k(x_prior, x_new)
        scale_tril_prior (float or Tensor): cholesky( k(x_prior, x_prior) + var_prior * I )
        covariance_new (float or Tensor): k(x_new, x_new) + var_post * I
    """

    arg_constraints = {"loc": constraints.real, "loc_prior": constraints.real_vector, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def mode(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(
        self,
        loc_new,
        loc_prior,
        y_prior,
        scale_tril_prior,  # k(x_prior, x_prior) + var_prior * I
        covariance_prior_new,  # k(x_prior, x_posterior)
        covariance_new,  # k(x_post, x_post) + var_post * I
        validate_args=None,
    ):
        batch_shape = torch.broadcast_shapes(
            loc_new.shape,
            loc_prior.shape[:-1],
            y_prior.shape[:-1],
            scale_tril_prior.shape[:-2],
            covariance_prior_new.shape[:-2],
            covariance_new.shape[:-2],
        )

        self.loc_prior = loc_prior.expand(batch_shape + (-1,))
        self.y_prior = y_prior.expand(batch_shape + (-1,))

        L_inv_Kxt = torch.linalg.solve_triangular(
            scale_tril_prior,
            covariance_prior_new,
            upper=False,  # [N_prior, 1]
        ).squeeze(-1)  # [N_prior]
        cor = covariance_new[..., 0, 0] - torch.pow(L_inv_Kxt, 2.0).sum(-1)  # empty shape
        L_post_cor = torch.sqrt(cor)  # empty shape

        epsi = y_prior - loc_prior
        L_inv_epsi = torch.linalg.solve_triangular(
            scale_tril_prior, epsi.unsqueeze(dim=-1), upper=False
        )  # [batch, N_prior, 1]
        self.loc = (
            loc_new.expand(batch_shape) + torch.matmul(L_inv_Kxt.unsqueeze(-2), L_inv_epsi)[..., 0, 0]
        )  # [batch,]
        self.scale = L_post_cor.expand(batch_shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GPMatheronUnivariate, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_prior_shape = batch_shape + self.loc_prior.shape[-1:]

        new.loc_prior = self.loc_prior.expand(loc_prior_shape)
        new.y_prior = self.y_prior.expand(loc_prior_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(GPMatheronUnivariate, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = self.scale**2
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        return -((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
