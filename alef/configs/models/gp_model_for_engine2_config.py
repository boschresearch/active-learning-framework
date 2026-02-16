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


class Engine2GPModelConfig(BasicGPModelConfig):
    optimize_hps: bool = False
    train_likelihood_variance: bool = False
    pertube_parameters_at_start: bool = False
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name = "Engine2GPModel"


class Engine2GPModelBEConfig(Engine2GPModelConfig):
    observation_noise: float = np.sqrt(0.003188044)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.822591987, 0.883898167, 7.234758056, 8.418512087],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine2GPModel_be"


class Engine2GPModelBENoContextConfig(Engine2GPModelBEConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.822591987, 0.883898167], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine2GPModel_be_no_context"


class Engine2GPModelTExConfig(Engine2GPModelConfig):
    observation_noise: float = np.sqrt(0.029146501)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[1.030446982, 4.113634137, 6.701675213, 6.82037554],
        base_variance=1,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine2GPModel_T_Ex"


class Engine2GPModelTExNoContextConfig(Engine2GPModelTExConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[1.030446982, 4.113634137], base_variance=1, input_dimension=2, add_prior=False
    )
    name = "Engine2GPModel_T_Ex_no_context"


class Engine2GPModelPI0vConfig(Engine2GPModelConfig):
    observation_noise: float = np.sqrt(0.110279469)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.259028483, 0.204969031, 0.87219542, 3.523564121],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine2GPModel_PI0v"


class Engine2GPModelPI0vNoContextConfig(Engine2GPModelPI0vConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.259028483, 0.204969031], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine2GPModel_PI0v_no_context"


class Engine2GPModelPI0sConfig(Engine2GPModelConfig):
    observation_noise: float = np.sqrt(0.375560725)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.552191418, 0.662460497, 1.451483663, 4.465192374],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine2GPModel_PI0s"


class Engine2GPModelPI0sNoContextConfig(Engine2GPModelPI0sConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.552191418, 0.662460497], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine2GPModel_PI0s_no_context"


class Engine2GPModelHCConfig(Engine2GPModelConfig):
    observation_noise: float = np.sqrt(0.11182418)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.506918606, 0.896232145, 2.256960119, 2.725814509],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine2GPModel_HC"


class Engine2GPModelHCNoContextConfig(Engine2GPModelHCConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[0.506918606, 0.896232145], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine2GPModel_HC_no_context"


class Engine2GPModelNOxConfig(Engine2GPModelConfig):
    observation_noise: float = np.sqrt(0.019885221)
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[1.339246015, 2.618131053, 3.865287007, 4.43784054],
        base_variance=1.0,
        input_dimension=4,
        add_prior=False,
    )
    name = "Engine2GPModel_NOx"


class Engine2GPModelNOxNoContextConfig(Engine2GPModelNOxConfig):
    kernel_config: BaseKernelConfig = Matern52WithPriorConfig(
        base_lengthscale=[1.339246015, 2.618131053], base_variance=1.0, input_dimension=2, add_prior=False
    )
    name = "Engine2GPModel_NOx_no_context"


if __name__ == "__main__":
    from alef.models.model_factory import ModelFactory

    m_config = Engine2GPModelBENoContextConfig()
    print(m_config.kernel_config)
    print(m_config.kernel_config.base_lengthscale)
    model = ModelFactory.build(m_config)

    # print_summary(model.kernel)
