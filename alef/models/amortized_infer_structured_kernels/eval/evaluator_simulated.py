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

import os

import numpy as np
from alef.configs.prior_parameters import NOISE_VARIANCE_EXPONENTIAL_LAMBDA
from alef.models.amortized_infer_structured_kernels.data_generators.dim_wise_additive_kernels_generator import (
    DimWiseAdditiveKernelGenerator,
)
from alef.models.base_model import BaseModel
from alef.configs.models.base_model_config import BaseModelConfig
from alef.models.amortized_infer_structured_kernels.data_generators.generator_factory import GeneratorFactory
from alef.models.amortized_infer_structured_kernels.config.data_generators.dim_wise_additive_generator_config import (
    BasicDimWiseAdditiveGeneratorConfig,
)
from alef.models.amortized_infer_structured_kernels.data_generators.simulator import SimulatedDataset, Simulator
from alef.models.amortized_infer_structured_kernels.gp.base_kernels import transform_kernel_list_to_expression
from alef.models.gp_model import GPModel
from alef.models.gp_model_amortized_ensemble import GPModelAmortizedEnsemble
from alef.models.gp_model_amortized_structured import GPModelAmortizedStructured
from alef.models.gp_model_pytorch import GPModelPytorch
from alef.models.model_factory import ModelFactory
from alef.utils.plotter import Plotter
from alef.utils.utils import calculate_rmse, write_dict_to_json
import time


class EvaluatorSimulated:
    def __init__(
        self,
        output_folder: str,
        generator_config: BasicDimWiseAdditiveGeneratorConfig,
        n_simulated: int,
        seed: int = 100,
    ):
        self.generator = GeneratorFactory.build(generator_config)
        assert isinstance(self.generator, DimWiseAdditiveKernelGenerator)
        self.generator_config = generator_config
        self.n_simulated = n_simulated
        self.output_folder = output_folder
        self.seed = seed
        self.do_plotting = False
        self.generator.set_initial_seed(self.seed)
        self.simulated_datasets = []
        self.only_gt_kernels = False
        self.fix_kernel = False
        self.generate_datasets()
        self.filter_datasets_for_extrapolation = False
        self.extrapolate_a = 0.2
        self.extrapolate_b = 0.8
        self.extrapolate_n_train = 100
        self.extrpolate_n_test = 100
        self.inverse_extrapolation = False

    def set_extrapolation(self, do_extrapolation, a, b, inverse, n_train, n_test):
        self.filter_datasets_for_extrapolation = do_extrapolation
        self.extrapolate_a = a
        self.extrapolate_b = b
        self.extrapolate_n_train = n_train
        self.extrpolate_n_test = n_test
        self.inverse_extrapolation = inverse

    def set_do_plotting(self, do_plotting):
        self.do_plotting = do_plotting

    def set_fix_kernel(self, fix_kernel):
        self.fix_kernel = fix_kernel
        assert not (self.only_gt_kernels and self.fix_kernel)

    def set_only_gt_kernels(self, only_gt_kernels):
        self.only_gt_kernels = only_gt_kernels
        assert not (self.only_gt_kernels and self.fix_kernel)

    def get_meta_data_dict(self):
        meta_data_dict = {}
        meta_data_dict["n_simulated"] = self.n_simulated
        meta_data_dict["generator_config"] = self.generator_config.dict()
        meta_data_dict["seed"] = self.seed
        meta_data_dict["only_gt_kernels"] = self.only_gt_kernels
        meta_data_dict["fix_kernel"] = self.fix_kernel

    def generate_datasets(self):
        for i in range(0, self.n_simulated):
            simulated_dataset = self.generator.generate_one_sample()
            self.simulated_datasets.append(simulated_dataset)

    def generate_custom_datasets(
        self, n_train, n_test, kernel_list, kernel_expression, box_lower, box_upper, sample_noise, noise_value=0.05
    ):
        self.simulated_datasets = []
        simulator = Simulator(box_lower, box_upper, NOISE_VARIANCE_EXPONENTIAL_LAMBDA)
        for i in range(0, self.n_simulated):
            simulated_dataset = simulator.create_sample(
                n_train, n_test, kernel_expression, sample_observation_noise=sample_noise, observation_noise=noise_value
            )
            simulated_dataset.add_kernel_list_gt(kernel_list)
            simulated_dataset.add_input_kernel_list(kernel_list)
            self.simulated_datasets.append(simulated_dataset)

    def evaluate_model(self, model_config: BaseModelConfig, model_name: str, model=None):
        write_dict_to_json(model_config.dict(), os.path.join(self.output_folder, model_name + "_model_config.json"))
        write_dict_to_json(
            self.get_meta_data_dict(), os.path.join(self.output_folder, model_name + "_simulator_config.json")
        )
        rmses = []
        nlls = []
        inference_times = []
        complete_times = []
        ns = []
        dimensions = []
        if model is None:
            model = ModelFactory.build(model_config)
        for simulated_data in self.simulated_datasets:
            assert isinstance(simulated_data, SimulatedDataset)
            # get associated input kernel
            if self.only_gt_kernels:
                kernel_list = simulated_data.get_kernel_list_gt()
            else:
                kernel_list = simulated_data.get_input_kernel_list()
            # set kernel in model
            if self.fix_kernel:
                model = self.update_input_dimension(model, model_config, simulated_data.get_input_dimension())
            else:
                model = self.set_kernel(model, model_config, kernel_list)

            # get train and test data
            x_data, y_data, x_test, y_test = self.get_datasets(simulated_data)

            # make inference
            time_before_infer = time.perf_counter()
            model.infer(x_data, y_data)
            time_after_infer = time.perf_counter()

            # do prediction
            pred_mu, pred_sigma = model.predictive_dist(x_test)
            time_after_pred = time.perf_counter()

            # calculate metrics
            rmse_model = calculate_rmse(pred_mu, y_test)
            nll_model = np.mean(-1 * model.predictive_log_likelihood(x_test, y_test))

            # store metrics
            ns.append(simulated_data.get_num_datapoints())
            dimensions.append(simulated_data.get_input_dimension())
            rmses.append(rmse_model)
            nlls.append(nll_model)
            inference_times.append(time_after_infer - time_before_infer)
            complete_times.append(time_after_pred - time_before_infer)
            if self.do_plotting:
                self.plot(x_data, y_data, x_test, y_test, model)

        np.savetxt(os.path.join(self.output_folder, f"rmse_{model_name}.csv"), np.array(rmses))
        np.savetxt(os.path.join(self.output_folder, f"nll_{model_name}.csv"), np.array(nlls))
        np.savetxt(os.path.join(self.output_folder, f"infer_times_{model_name}.csv"), np.array(inference_times))
        np.savetxt(os.path.join(self.output_folder, f"times_{model_name}.csv"), np.array(complete_times))
        np.savetxt(os.path.join(self.output_folder, f"ns_{model_name}.csv"), np.array(ns))
        np.savetxt(os.path.join(self.output_folder, f"ds_{model_name}.csv"), np.array(dimensions))

    def get_datasets(self, simulated_data):
        x_data, y_data = simulated_data.get_dataset()
        x_test, y_test = simulated_data.get_test_dataset()
        if self.filter_datasets_for_extrapolation:
            return self.filter_data(x_data, y_data, x_test, y_test)
        return x_data, y_data, x_test, y_test

    def filter_data(self, x_data, y_data, x_test, y_test):
        if self.inverse_extrapolation:
            data_filter = np.any((x_data < self.extrapolate_a) | (x_data > self.extrapolate_b), axis=1)
        else:
            data_filter = np.all((x_data >= self.extrapolate_a) & (x_data <= self.extrapolate_b), axis=1)
        filtered_x_data = x_data[data_filter][: self.extrapolate_n_train]
        filtered_y_data = y_data[data_filter][: self.extrapolate_n_train]
        if self.inverse_extrapolation:
            test_filter = np.all((x_test >= self.extrapolate_a) & (x_test <= self.extrapolate_b), axis=1)
        else:
            test_filter = np.any((x_test < self.extrapolate_a) | (x_test > self.extrapolate_b), axis=1)
        filtered_x_test = x_test[test_filter][: self.extrpolate_n_test]
        filtered_y_test = y_test[test_filter][: self.extrpolate_n_test]
        return filtered_x_data, filtered_y_data, filtered_x_test, filtered_y_test

    def plot(self, x_data, y_data, x_test, y_test, model: BaseModel):
        if x_data.shape[1] == 1:
            bound_a = min(np.min(x_data), np.min(x_test))
            bound_b = max(np.max(x_data), np.max(x_test))
            x_grid = np.expand_dims(np.linspace(bound_a - 0.05, bound_b + 0.05, 400), axis=1)
            pred_mu_grid, pred_sigma_grid = model.predictive_dist(x_grid)
            plotter = Plotter(1)
            plotter.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid), np.squeeze(pred_sigma_grid), 0)
            plotter.add_datapoints(x_data, y_data, "green", 0)
            plotter.add_datapoints(x_test, y_test, "red", 0)
            plotter.show()

    def set_kernel(self, model, model_config, kernel_list):
        if isinstance(model, GPModelPytorch):
            model = ModelFactory.build(model_config)
            kernel_expression = transform_kernel_list_to_expression(kernel_list, add_prior=False)
            model.kernel_module = kernel_expression.get_kernel()
        elif isinstance(model, GPModelAmortizedStructured):
            model.clear_cache()
            model.set_kernel_list(kernel_list)
        elif isinstance(model, GPModel):
            model = ModelFactory.build(model_config)
            kernel_expression = transform_kernel_list_to_expression(kernel_list, add_prior=False, use_gpflow=True)
            model.set_kernel(kernel_expression.get_kernel())
        elif isinstance(model, GPModelAmortizedEnsemble):
            pass
        else:
            raise NotImplementedError
        return model

    def update_input_dimension(self, model, model_config, input_dimension):
        new_model_config = ModelFactory.change_input_dimension(model_config, input_dimension)
        if isinstance(model, GPModelPytorch):
            model = ModelFactory.build(new_model_config)
        elif isinstance(model, GPModelAmortizedStructured):
            pass
        elif isinstance(model, GPModel):
            model = ModelFactory.build(new_model_config)
        elif isinstance(model, GPModelAmortizedEnsemble):
            pass
        else:
            raise NotImplementedError
        return model

    def load_arrays(self, model_name):
        rmses = np.loadtxt(os.path.join(self.output_folder, f"rmse_{model_name}.csv"))
        nlls = np.loadtxt(os.path.join(self.output_folder, f"nll_{model_name}.csv"))
        infer_times = np.loadtxt(os.path.join(self.output_folder, f"infer_times_{model_name}.csv"))
        times = np.loadtxt(os.path.join(self.output_folder, f"times_{model_name}.csv"))
        return rmses, nlls, infer_times, times
