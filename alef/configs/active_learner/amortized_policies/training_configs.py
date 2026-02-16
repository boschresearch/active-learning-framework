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

from typing import Dict, Union
from pydantic import BaseSettings

import torch

from .loss_configs import BasicAmortizedPolicyLossConfig, BasicLossCurriculumConfig
from .policy_configs import BaseAmortizedPolicyConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig

__all__ = [
    "AmortizedNonmyopicContinuousFixGPPolicyTrainingConfig",
    "AmortizedNonmyopicContinuousRandomGPPolicyTrainingConfig",
    "AmortizedNonmyopicContinuousRandomGPLargerLRPolicyTrainingConfig",
]


class BaseAmortizedPolicyTrainingConfig(BaseSettings):
    optimizer: torch.optim.Optimizer = torch.optim.RAdam  # torch.optim.Adam
    optim_args: Dict = {"lr": 1e-4}  # {"lr": 1e-5, "betas": [0.9, 0.999], "weight_decay": 0,}
    gamma: float = 0.98
    policy_config: BaseAmortizedPolicyConfig
    loss_config: Union[BasicAmortizedPolicyLossConfig, BasicLossCurriculumConfig]


class AmortizedNonmyopicContinuousFixGPPolicyTrainingConfig(BaseAmortizedPolicyTrainingConfig):
    kernel_config: BaseKernelPytorchConfig
    n_steps: int
    sample_gp_prior: bool = False
    optim_args: Dict = {"lr": 1e-4}  # {"lr": 1e-5, "betas": [0.9, 0.999], "weight_decay": 0,}
    gamma: float = 0.98


class AmortizedNonmyopicContinuousRandomGPPolicyTrainingConfig(BaseAmortizedPolicyTrainingConfig):
    kernel_config: BaseKernelPytorchConfig
    n_steps: int
    sample_gp_prior: bool = True
    optim_args: Dict = {"lr": 1e-4}
    gamma: float = 0.98


class AmortizedNonmyopicContinuousRandomGPLargerLRPolicyTrainingConfig(
    AmortizedNonmyopicContinuousRandomGPPolicyTrainingConfig
):
    optim_args: Dict = {"lr": 1e-3}
