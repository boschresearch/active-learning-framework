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

from typing import List
from pydantic import BaseSettings


__all__ = [
    "ContinuousDADLossConfig",
    "ContinuousScoreDADLossConfig",
    "GPEntropyLoss1Config",
    "GPEntropyLoss2Config",
    "GPMILoss1Config",
    "GPMILoss2Config",
    "GPMI_EntropyLoss1Config",
    "GPMI_EntropyLoss2Config",
]


class BasicAmortizedPolicyLossConfig(BaseSettings):
    batch_size: int
    num_kernels: int
    num_functions_per_kernel: int
    num_epochs: int = 800
    epochs_size: int = 50
    name: str = "basic_amortized_policy_loss"


class BasicLossCurriculumConfig(BaseSettings):
    loss_config_list: List[BasicAmortizedPolicyLossConfig]
    name: str = "basic_amortized_policy_curriculum"


############
# DAD loss
#
class ContinuousDADLossConfig(BasicAmortizedPolicyLossConfig):
    batch_size: int = 100
    num_kernels: int = 100
    num_functions_per_kernel: int = 100
    name: str = "dad_loss"


class ContinuousScoreDADLossConfig(BasicAmortizedPolicyLossConfig):
    batch_size: int = 100
    num_kernels: int = 100
    num_functions_per_kernel: int = 100
    name: str = "dad_reinforce_loss"


############
# GP Entropy loss
#
class GPEntropyLoss1Config(BasicAmortizedPolicyLossConfig):
    batch_size: int = 20
    num_kernels: int = 25
    num_functions_per_kernel: int = 25
    name: str = "gp_entropy_loss1"


class GPEntropyLoss2Config(BasicAmortizedPolicyLossConfig):
    batch_size: int = 20
    num_kernels: int = 25
    num_functions_per_kernel: int = 25
    name: str = "gp_entropy_loss2"


############
# GP Mutual Information loss
#
class GPMILoss1Config(BasicAmortizedPolicyLossConfig):
    batch_size: int = 20  # 1 GPU does not have enough memory
    num_kernels: int = 25
    num_functions_per_kernel: int = 25
    num_epochs: int = 400
    name: str = "gp_mi_loss1"


class GPMILoss2Config(BasicAmortizedPolicyLossConfig):
    batch_size: int = 20  # 1 GPU does not have enough memory
    num_kernels: int = 25
    num_functions_per_kernel: int = 25
    num_epochs: int = 400
    name: str = "gp_mi_loss2"


############
# Curriculum of losses
#
class GPMI_EntropyLoss1Config(BasicLossCurriculumConfig):
    loss_config_list: List[BasicAmortizedPolicyLossConfig] = [
        GPMILoss1Config(num_epochs=150),
        GPEntropyLoss1Config(num_epochs=450),
    ]
    name: str = "gp_mi_and_entropy_curriculum1"


class GPMI_EntropyLoss2Config(BasicLossCurriculumConfig):
    loss_config_list: List[BasicAmortizedPolicyLossConfig] = [
        GPMILoss2Config(num_epochs=150),
        GPEntropyLoss2Config(num_epochs=450),
    ]
    name: str = "gp_mi_and_entropy_curriculum2"
