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

import torch

import pyro
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item

from .base_oed_loss import BaseOEDLoss
from .base_multiple_steps_gp_loss import _GPRMSEComputer

"""
The following code is adapted from
https://github.com/ae-foster/dad/blob/main/contrastive/mi.py
Copyright (c) 2021 Adam Foster and Desi R. Ivanova, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).

and

https://github.com/desi-ivanova/idad/blob/main/estimators/mi.py
Copyright (c) 2021 Adam Foster, Desi R. Ivanova, Steven Kleinegesse, licensed under the MIT License,
cf. LICENSE file in the root directory of this source tree).
"""


class MutualInformation(BaseOEDLoss):
    def __init__(self, batch_size, num_kernels, num_functions_per_kernel, data_source=None, name=None):
        super().__init__(batch_size, data_source=data_source, name=name)
        self.num_kernels = num_kernels
        self.num_functions_per_kernel = num_functions_per_kernel
        self.test_computer = _GPRMSEComputer()

    def get_primary_rollout(self, process, graph_type="flat", detach=False):
        trace = self.get_rollout(
            process,
            tuple(),
            {"batch_size": self.batch_size, "num_kernels": 1, "num_functions": 1},
            "outer_vectorization",
            graph_type,
            detach,
        )
        trace.compute_log_prob()
        return trace

    def validation(self, process):
        """
        compute the validation values which does not need to be differentiated (for visualization, analysis etc)

        return tuple of 2 floats, the mean and standard error of the validation values
        """
        B = min(self.batch_size, 10)
        Nk = min(self.num_kernels, 10)
        Nf = min(self.num_functions_per_kernel, 10)

        test_trace = self.get_test_rollout(
            process,
            (),
            {"batch_size": B, "num_kernels": Nk, "num_functions": Nf, "num_test_points": 200},
            "test_vectorization",
            detach=True,
        )
        rmse = self.test_computer(*test_trace.nodes["_RETURN"]["value"])  # [B, Nk, Nf]
        rmse_flatten = rmse.flatten()  # [B * Nk * Nf]

        rmse_mean = rmse_flatten.mean()
        rmse_stderr = torch.sqrt(rmse_flatten.var() / rmse_flatten.shape[0])
        return rmse_mean, rmse_stderr


class PriorContrastiveEstimation(MutualInformation):
    def compute_observation_log_prob(self, trace):
        """
        Computes the log probability of observations given latent variables and designs.
        :param trace: a Pyro trace object
        :return: the log prob tensor
        """
        return sum(node["log_prob"] for node in trace.nodes.values() if node.get("subtype") == "observation_sample")

    def get_contrastive_rollout(
        self,
        process,
        trace,
        graph_type="flat",
        detach=False,
    ):
        conditions = {
            name: node["value"].expand(
                node["value"].shape[:1]
                + (self.num_kernels, self.num_functions_per_kernel)
                + node["value"].shape[
                    3:
                ]  # the first two dim of primary trace are 1, 1 (place holder of num_kernels, num_funcs)
            )
            for name, node in trace.nodes.items()
            if node.get("subtype") in ["observation_sample", "design_sample"]
        }
        conditional_model = pyro.condition(
            self._vectorized(process.process, name="inner_vectorization"), data=conditions
        )
        trace = self.get_rollout_from_pyro_model(
            conditional_model,
            tuple(),
            {
                "batch_size": self.batch_size,
                "num_kernels": self.num_kernels,
                "num_functions": self.num_functions_per_kernel,
            },
            graph_type=graph_type,
            detach=detach,
        )
        trace.compute_log_prob()
        return trace

    def differentiable_loss(self, process):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(process)
        contrastive_trace = self.get_contrastive_rollout(process, primary_trace)

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)  # [B, 1, 1]
        obs_log_prob_contrastive = self.compute_observation_log_prob(
            contrastive_trace
        )  # [B, num_kernels, num_functions]
        obs_log_prob_combined = torch.cat(
            [
                obs_log_prob_primary.flatten(1, 2),  # [B, 1]
                obs_log_prob_contrastive.flatten(1, 2),  # [B, num_kernels * num_functions]
            ],
            dim=1,
        ).logsumexp(1)  # log sum exp over different prior functions

        loss = (obs_log_prob_combined - obs_log_prob_primary.flatten()).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, process, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation; == -EIG proxy
        """
        loss_to_constant = torch_item(self.differentiable_loss(process, *args, **kwargs))
        return loss_to_constant - math.log(self.num_kernels * self.num_functions_per_kernel + 1)


class NestedMonteCarloEstimation(PriorContrastiveEstimation):
    def differentiable_loss(self, process):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        primary_trace = self.get_primary_rollout(process)
        contrastive_trace = self.get_contrastive_rollout(process, primary_trace)

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)  # [B, 1, 1]
        obs_log_prob_contrastive = self.compute_observation_log_prob(
            contrastive_trace
        )  # [B, num_kernels, num_functions]
        obs_log_prob_combined = obs_log_prob_contrastive.flatten(1, 2).logsumexp(1)
        loss = (obs_log_prob_combined - obs_log_prob_primary.flatten()).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, process, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation
        """
        loss_to_constant = torch_item(self.differentiable_loss(process, *args, **kwargs))
        return loss_to_constant - math.log(self.num_kernels * self.num_functions_per_kernel)


class PriorContrastiveEstimationScoreGradient(PriorContrastiveEstimation):
    def differentiable_loss(self, process):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        -> log p(h_T|theta, pi_phi)* const(g_phi) + g_phi
        """
        primary_trace = self.get_primary_rollout(process)
        contrastive_trace = self.get_contrastive_rollout(process, primary_trace)

        obs_log_prob_primary = self.compute_observation_log_prob(primary_trace)  # [B, 1, 1]
        obs_log_prob_contrastive = self.compute_observation_log_prob(
            contrastive_trace
        )  # [B, num_kernels, num_functions]

        obs_log_prob_combined = torch.cat(
            [
                obs_log_prob_primary.flatten(1, 2),  # [B, 1]
                obs_log_prob_contrastive.flatten(1, 2),  # [B, num_kernels * num_functions]
            ],
            dim=1,
        ).logsumexp(1)  # log sum exp over different prior functions

        with torch.no_grad():
            g_no_grad = obs_log_prob_primary.flatten() - obs_log_prob_combined

        loss = -(g_no_grad * obs_log_prob_primary.flatten() - obs_log_prob_combined).mean(0)

        warn_if_nan(loss, "loss")
        return loss

    def loss(self, process, *args, **kwargs):
        """
        :returns: returns an estimate of the mutual information
        :rtype: float
        Evaluates the MI lower bound using prior contrastive estimation
        """
        basic_pce = super().differentiable_loss(process, *args, **kwargs)
        loss_to_constant = torch_item(basic_pce)
        loss = loss_to_constant - math.log(self.num_kernels * self.num_functions_per_kernel + 1)
        return loss
