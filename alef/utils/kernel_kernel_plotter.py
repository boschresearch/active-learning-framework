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

import logging
import os
from typing import List
from matplotlib import pyplot as plt
import matplotlib

import numpy as np
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.models.object_gp_model_config import BasicObjectGPModelConfig
from alef.kernels.kernel_grammar.kernel_grammar import BaseKernelGrammarExpression
from alef.kernels.kernel_kernel_grammar_tree import OptimalTransportKernelKernel
from alef.kernels.kernel_kernel_hellinger import KernelKernelHellinger
from alef.models.model_factory import ModelFactory
from alef.models.object_mean_functions import ObjectConstant
from alef.utils.utils import manhatten_distance

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)


def set_font_sizes(font_size, only_axis=True):
    if only_axis:
        matplotlib.rc("xtick", labelsize=font_size)
        matplotlib.rc("ytick", labelsize=font_size)
    else:
        font = {"family": "normal", "size": font_size}

        matplotlib.rc("font", **font)


class KernelKernelPlotter:
    def __init__(self, kernel_kernel_config: BaseKernelConfig, output_path: str):
        self.kernel_kernel_config = kernel_kernel_config
        self.output_path = output_path

    def target_difference_distance_plot(
        self, x_data: List[BaseKernelGrammarExpression], y_data: np.array, target_name: str, data_set_identifier: str
    ):
        object_gp_model_config = BasicObjectGPModelConfig(
            kernel_config=self.kernel_kernel_config, perform_multi_start_optimization=False, optimize_hps=True
        )
        model = ModelFactory.build(object_gp_model_config)
        assert isinstance(model.kernel, OptimalTransportKernelKernel) or isinstance(model.kernel, KernelKernelHellinger)
        model.set_mean_function(ObjectConstant())
        model.infer(x_data, y_data)
        self.save_kernel_parameters(model, data_set_identifier)
        manhatten_distances = model.model.kernel.get_manhatten_distances(x_data)
        target_abs_difference = manhatten_distance(y_data, y_data)
        n_data = len(x_data)
        for i, distance_matrix in enumerate(manhatten_distances):
            self.create_distance_matrix_plots(
                n_data, target_name, distance_matrix, target_abs_difference, data_set_identifier, "W" + str(i)
            )
        distance_matrix = model.model.kernel.get_distance_matrix(x_data)
        self.create_distance_matrix_plots(
            n_data, target_name, distance_matrix, target_abs_difference, data_set_identifier, ""
        )

    def create_distance_matrix_plots(
        self, n_data, target_name, distance_matrix, target_abs_difference, data_set_identifier, suffix
    ):
        distance_list = []
        difference_list = []
        for i in range(0, n_data):
            for j in range(i + 1, n_data):
                distance_list.append(distance_matrix[i, j])
                difference_list.append(target_abs_difference[i, j])
        distances = np.array(distance_list)
        differences = np.array(difference_list)
        indexes = np.arange(0, len(distances))
        indexes = np.random.choice(indexes, 1500, replace=False)
        _, ax = plt.subplots(1, 1)
        ax.plot(distances[indexes], differences[indexes], "x", color="blue")
        ax.set_ylabel(target_name + " Difference")
        ax.set_xlabel("SOT - Distance")
        ax.set_title("SOT vs. " + target_name)
        plt.savefig(os.path.join(self.output_path, "sot_vs_diff_" + suffix + "_" + data_set_identifier + ".png"))
        plt.close()
        # ax.set_xlim((0.0, 3.05))
        _, ax2 = plt.subplots(1, 1)
        im2 = ax2.imshow(distance_matrix)
        ax2.set_xlabel("Kernel Index")
        ax2.set_ylabel("Kernel Index")
        ax2.set_title("SOT Distance Matrix")
        plt.colorbar(im2, ax=ax2)
        plt.savefig(
            os.path.join(self.output_path, "sot_dist_matrix_" + suffix + "_" + data_set_identifier + ".png"), dpi=400
        )
        plt.close()

    def save_kernel_parameters(self, model, data_set_identifier):
        if isinstance(model.model.kernel, OptimalTransportKernelKernel):
            alphas = model.model.kernel.alphas.numpy()
            lengthscale = model.model.kernel.lengthscale.numpy()
            variance = model.model.kernel.variance.numpy()
            likelihood_variance = model.model.likelihood.variance.numpy()
            parameter_vector = np.concatenate((alphas, [lengthscale], [variance], [likelihood_variance]))
            np.savetxt(
                os.path.join(self.output_path, "kernel_parameters_" + data_set_identifier + ".txt"), parameter_vector
            )
