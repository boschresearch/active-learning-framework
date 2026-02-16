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

import numpy as np
from alef.configs.models.gp_model_config import BasicGPModelConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.matern52_configs import Matern52WithPriorConfig
from alef.enums.global_model_enums import PredictionQuantity
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE


class Engine1GPModelConfig(BasicGPModelConfig):
    optimize_hps: bool = False
    train_likelihood_variance: bool = False
    pertube_parameters_at_start: bool = False
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name = "Engine1GPModel"


class Engine1GPModelBEConfig(Engine1GPModelConfig):
    observation_noise: float = np.sqrt(0.004175437)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[1.313312213, 0.446024226, 33.09974174, 8.031606669],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine1GPModel_be"


class Engine1GPModelBENoContextConfig(Engine1GPModelBEConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[1.313312213, 0.446024226], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine1GPModel_be_no_context"


class Engine1GPModelTExConfig(Engine1GPModelConfig):
    observation_noise: float = np.sqrt(0.003531304)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[2.283112178, 4.269472684, 18.46178265, 14.06234037],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine1GPModel_T_Ex"


class Engine1GPModelTExNoContextConfig(Engine1GPModelTExConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[2.283112178, 4.269472684], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine1GPModel_T_Ex_no_context"


class Engine1GPModelPI0vConfig(Engine1GPModelConfig):
    observation_noise: float = np.sqrt(0.065157721)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.515865489, 0.997737456, 3.642025088, 8.085118276],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine1GPModel_PI0v"


class Engine1GPModelPI0vNoContextConfig(Engine1GPModelPI0vConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.515865489, 0.997737456], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine1GPModel_PI0v_no_context"


class Engine1GPModelPI0sConfig(Engine1GPModelConfig):
    observation_noise: float = np.sqrt(0.215896779)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.498724796, 1.186691392, 3.098206541, 6.275467397],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine1GPModel_PI0s"


class Engine1GPModelPI0sNoContextConfig(Engine1GPModelPI0sConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.498724796, 1.186691392], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine1GPModel_PI0s_no_context"


class Engine1GPModelHCConfig(Engine1GPModelConfig):
    observation_noise: float = np.sqrt(0.043576866)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.652439592, 1.133571675, 2.401624557, 4.075449211],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine1GPModel_HC"


class Engine1GPModelHCNoContextConfig(Engine1GPModelHCConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.652439592, 1.133571675], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine1GPModel_HC_no_context"


class Engine1GPModelNOxConfig(Engine1GPModelConfig):
    observation_noise: float = np.sqrt(0.005284851)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.510919258, 2.106157995, 4.538927013, 4.112122696],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine1GPModel_NOx"


class Engine1GPModelNOxNoContextConfig(Engine1GPModelNOxConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.510919258, 2.106157995], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine1GPModel_NOx_no_context"


if __name__ == "__main__":
    from alef.models.model_factory import ModelFactory
    from gpflow.utilities import print_summary

    m_config = Engine1GPModelBENoContextConfig()
    model = ModelFactory.build(m_config)

    print_summary(model.kernel)
