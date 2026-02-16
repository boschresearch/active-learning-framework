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
from typing import Sequence
from alef.configs.models.mogp_model_so_config import BasicSOMOGPModelConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.multi_output_kernels.coregionalization_Platent_kernel_configs import (
    BasicCoregionalizationPLConfig,
)


class EngineMOGPModelConfig(BasicSOMOGPModelConfig):
    optimize_hps: bool = False
    name = "EngineMOGPModel"


class EngineMOGPModelBEConfig(EngineMOGPModelConfig):
    observation_noise: Sequence[float] = [np.sqrt(0.003067655), np.sqrt(0.002456222)]
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=4,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[86.449656, 73.228827], [92.504874, 79.411917]]),
            "kernels[0].kernels[1].kappa": np.array([6.189183e-12, 1.800133e-07]),
            "kernels[0].kernels[0].lengthscales": np.array([13.832321, 4.681651, 170.819745, 99.383643]),
            "kernels[1].kernels[1].W": np.array([[0.027161, 0.050456], [-0.107991, 0.095041]]),
            "kernels[1].kernels[1].kappa": np.array([1.849854e-12, 2.519246e-05]),
            "kernels[1].kernels[0].lengthscales": np.array([0.221188, 0.597982, 1.580639, 3.393215]),
        },
    )
    name = "EngineMOGPModel_be"


class EngineMOGPModelBENoContextConfig(EngineMOGPModelBEConfig):
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=2,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[86.449656, 73.228827], [92.504874, 79.411917]]),
            "kernels[0].kernels[1].kappa": np.array([6.189183e-12, 1.800133e-07]),
            "kernels[0].kernels[0].lengthscales": np.array([13.832321, 4.681651]),
            "kernels[1].kernels[1].W": np.array([[0.027161, 0.050456], [-0.107991, 0.095041]]),
            "kernels[1].kernels[1].kappa": np.array([1.849854e-12, 2.519246e-05]),
            "kernels[1].kernels[0].lengthscales": np.array([0.221188, 0.597982]),
        },
    )
    name = "EngineMOGPModel_be_no_context"


class EngineMOGPModelTExConfig(EngineMOGPModelConfig):
    observation_noise: Sequence[float] = [np.sqrt(0.001188703), np.sqrt(0.008943092)]
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=4,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[0.002475, -0.045805], [0.023424, -0.154974]]),
            "kernels[0].kernels[1].kappa": np.array([3.451602e-05, 1.914297e-03]),
            "kernels[0].kernels[0].lengthscales": np.array([0.005694, 1.721599, 0.01774, 1.315215]),
            "kernels[1].kernels[1].W": np.array([[1.22473, 1.009937], [1.571536, 0.812066]]),
            "kernels[1].kernels[1].kappa": np.array([2.348218e-05, 2.013816e-02]),
            "kernels[1].kernels[0].lengthscales": np.array([2.774383, 4.661028, 25.229363, 18.295844]),
        },
    )
    name = "EngineMOGPModel_T_Ex"


class EngineMOGPModelTExNoContextConfig(EngineMOGPModelTExConfig):
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=2,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[0.002475, -0.045805], [0.023424, -0.154974]]),
            "kernels[0].kernels[1].kappa": np.array([3.451602e-05, 1.914297e-03]),
            "kernels[0].kernels[0].lengthscales": np.array([0.005694, 1.721599]),
            "kernels[1].kernels[1].W": np.array([[1.22473, 1.009937], [1.571536, 0.812066]]),
            "kernels[1].kernels[1].kappa": np.array([2.348218e-05, 2.013816e-02]),
            "kernels[1].kernels[0].lengthscales": np.array([2.774383, 4.661028]),
        },
    )
    name = "EngineMOGPModel_T_Ex_no_context"


class EngineMOGPModelPI0vConfig(EngineMOGPModelConfig):
    observation_noise: Sequence[float] = [np.sqrt(0.088651085), np.sqrt(0.109771388)]
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=4,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[-0.268319, 2.021842], [-1.072279, 1.078857]]),
            "kernels[0].kernels[1].kappa": np.array([0.002676, 0.005255]),
            "kernels[0].kernels[0].lengthscales": np.array([1.335698, 2.375447, 5.487522, 29.874067]),
            "kernels[1].kernels[1].W": np.array([[0.193601, 0.092708], [0.527937, 0.252555]]),
            "kernels[1].kernels[1].kappa": np.array([7.038502e-07, 1.894363e-04]),
            "kernels[1].kernels[0].lengthscales": np.array([0.025244, 2.637916, 0.204546, 2.70691]),
        },
    )
    name = "EngineMOGPModel_PI0v"


class EngineMOGPModelPI0vNoContextConfig(EngineMOGPModelPI0vConfig):
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=2,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[-0.268319, 2.021842], [-1.072279, 1.078857]]),
            "kernels[0].kernels[1].kappa": np.array([0.002676, 0.005255]),
            "kernels[0].kernels[0].lengthscales": np.array([1.335698, 2.375447]),
            "kernels[1].kernels[1].W": np.array([[0.193601, 0.092708], [0.527937, 0.252555]]),
            "kernels[1].kernels[1].kappa": np.array([7.038502e-07, 1.894363e-04]),
            "kernels[1].kernels[0].lengthscales": np.array([0.025244, 2.637916]),
        },
    )
    name = "EngineMOGPModel_PI0v_no_context"


class EngineMOGPModelPI0sConfig(EngineMOGPModelConfig):
    observation_noise: Sequence[float] = [np.sqrt(1.00199e-06), np.sqrt(0.020707731)]
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=4,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[0.174823, 0.412049], [0.254444, 0.413127]]),
            "kernels[0].kernels[1].kappa": np.array([7.708159e-03, 7.846002e-06]),
            "kernels[0].kernels[0].lengthscales": np.array([1.897227e-04, 2.555478e00, 1.010545e-01, 1.421824e00]),
            "kernels[1].kernels[1].W": np.array([[0.902639, 0.624434], [0.492819, 0.58611]]),
            "kernels[1].kernels[1].kappa": np.array([1.582385e-04, 2.253248e-01]),
            "kernels[1].kernels[0].lengthscales": np.array([0.799164, 1.353611, 3.420253, 11.974183]),
        },
    )
    name = "EngineMOGPModel_PI0s"


class EngineMOGPModelPI0sNoContextConfig(EngineMOGPModelPI0sConfig):
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=2,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[0.174823, 0.412049], [0.254444, 0.413127]]),
            "kernels[0].kernels[1].kappa": np.array([7.708159e-03, 7.846002e-06]),
            "kernels[0].kernels[0].lengthscales": np.array([1.897227e-04, 2.555478e00]),
            "kernels[1].kernels[1].W": np.array([[0.902639, 0.624434], [0.492819, 0.58611]]),
            "kernels[1].kernels[1].kappa": np.array([1.582385e-04, 2.253248e-01]),
            "kernels[1].kernels[0].lengthscales": np.array([0.799164, 1.353611]),
        },
    )
    name = "EngineMOGPModel_PI0s_no_context"


class EngineMOGPModelHCConfig(EngineMOGPModelConfig):
    observation_noise: Sequence[float] = [np.sqrt(0.005836176), np.sqrt(2.23351e-06)]
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=4,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[0.763583, -0.995621], [1.283384, -0.05619]]),
            "kernels[0].kernels[1].kappa": np.array([0.024953, 0.071363]),
            "kernels[0].kernels[0].lengthscales": np.array([0.852237, 2.030005, 3.119074, 7.024314]),
            "kernels[1].kernels[1].W": np.array([[0.2738, -0.254895], [-0.408654, -0.260024]]),
            "kernels[1].kernels[1].kappa": np.array([1.497842e-12, 4.890318e-12]),
            "kernels[1].kernels[0].lengthscales": np.array([0.141137, 3.950036, 0.006168, 0.352251]),
        },
    )
    name = "EngineMOGPModel_HC"


class EngineMOGPModelHCNoContextConfig(EngineMOGPModelHCConfig):
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=2,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[0.763583, -0.995621], [1.283384, -0.05619]]),
            "kernels[0].kernels[1].kappa": np.array([0.024953, 0.071363]),
            "kernels[0].kernels[0].lengthscales": np.array([0.852237, 2.030005]),
            "kernels[1].kernels[1].W": np.array([[0.2738, -0.254895], [-0.408654, -0.260024]]),
            "kernels[1].kernels[1].kappa": np.array([1.497842e-12, 4.890318e-12]),
            "kernels[1].kernels[0].lengthscales": np.array([0.141137, 3.950036]),
        },
    )
    name = "EngineMOGPModel_HC_no_context"


class EngineMOGPModelNOxConfig(EngineMOGPModelConfig):
    observation_noise: Sequence[float] = [np.sqrt(1.00296e-06), np.sqrt(0.039309905)]
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=4,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[1.821656, -0.759334], [1.10534, -1.119072]]),
            "kernels[0].kernels[1].kappa": np.array([0.038327, 0.027053]),
            "kernels[0].kernels[0].lengthscales": np.array([0.754963, 3.620241, 4.359479, 9.137893]),
            "kernels[1].kernels[1].W": np.array([[-0.131281, 0.047761], [-0.026116, 0.009508]]),
            "kernels[1].kernels[1].kappa": np.array([5.630268e-05, 7.009039e-14]),
            "kernels[1].kernels[0].lengthscales": np.array([1.580137e-01, 1.989496e-03, 2.376370e00, 1.756282e00]),
        },
    )
    name = "EngineMOGPModel_NOx"


class EngineMOGPModelNOxNoContextConfig(EngineMOGPModelNOxConfig):
    # observation_noise: float = np.sqrt([])
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=2,
        output_dimension=2,
        fix_kernel=True,
        assign_values=True,
        parameter_values={
            "kernels[0].kernels[1].W": np.array([[1.821656, -0.759334], [1.10534, -1.119072]]),
            "kernels[0].kernels[1].kappa": np.array([0.038327, 0.027053]),
            "kernels[0].kernels[0].lengthscales": np.array([0.754963, 3.620241]),
            "kernels[1].kernels[1].W": np.array([[-0.131281, 0.047761], [-0.026116, 0.009508]]),
            "kernels[1].kernels[1].kappa": np.array([5.630268e-05, 7.009039e-14]),
            "kernels[1].kernels[0].lengthscales": np.array([1.580137e-01, 1.989496e-03]),
        },
    )
    name = "EngineMOGPModel_NOx_no_context"


if __name__ == "__main__":
    from alef.models.model_factory import ModelFactory
    from gpflow.utilities import print_summary

    m_config = EngineMOGPModelBEConfig()
    print(m_config.kernel_config)
    print(m_config.kernel_config.base_lengthscale)
    model = ModelFactory.build(m_config)

    print_summary(model.kernel)
