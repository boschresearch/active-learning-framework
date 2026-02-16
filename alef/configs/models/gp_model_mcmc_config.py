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

from alef.configs.models.base_model_config import BaseModelConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.models.gp_model_mcmc import InternalInferenceType
from typing import List
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE


class BasicGPModelMCMCConfig(BaseModelConfig):
    input_dimension: int
    n_samples: int = 5
    n_burnin: int = 20
    n_thinned: int = 5
    train_likelihood_variance: bool = True
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    initial_observation_noise: float = 0.01
    internal_inference_type: InternalInferenceType = InternalInferenceType.MAP
    perform_multi_start_opt_in_each_step: bool = False


class AdditiveProposalGPModelMCMCConfig(BasicGPModelMCMCConfig):
    initial_partition_list: List
    name = "AdditiveProposalGPModelMCMC"


class KernelGrammarGPModelMCMCConfig(BasicGPModelMCMCConfig):
    initial_base_kernel_config: BaseKernelConfig
    add_hp_prior: bool
    name = "KernelGrammarGPModelMCMC"


if __name__ == "__main__":
    AdditiveProposalGPModelMCMCConfig(input_dimension=2, initial_partition_list=[[0, 1], [2]])
