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
from alef.configs.kernels.rbf_configs import BasicRBFConfig
from alef.configs.models.gp_model_config import GPModelFastConfig
from alef.configs.models.gp_model_pytorch_ensemble_config import BasicGPModelPytorchEnsembleConfig
from alef.configs.models.svgp_model_pytorch_config import BasicSVGPModelPytorchConfig
from alef.data_sets.base_data_set import BaseDataset
from alef.data_sets.airfoil import Airfoil
from alef.data_sets.power_plant import PowerPlant
from alef.data_sets.energy import Energy
from alef.data_sets.airline_passenger import AirlinePassenger
from alef.data_sets.wine import Wine
from alef.data_sets.yacht import Yacht
from alef.data_sets.concrete import Concrete
from alef.configs.models.gp_model_pytorch_config import BasicGPModelPytorchConfig, GPModelPytorchMultistartConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicRBFPytorchConfig,
    RBFWithPriorPytorchConfig,
)
from alef.enums.global_model_enums import PredictionQuantity
from alef.models.amortized_infer_structured_kernels.gp.base_kernels import (
    BaseKernelTypes,
    transform_kernel_list_to_expression,
)
import numpy as np
from alef.models.gp_model_amortized_ensemble import GPModelAmortizedEnsemble

from alef.models.gp_model_amortized_structured import GPModelAmortizedStructured
from alef.models.model_factory import ModelFactory
from alef.utils.plot_utils import create_box_plot_from_dict
from alef.utils.utils import calculate_rmse, load_pickled_object, pickle_object
import time

import gpytorch

gpytorch.settings.lazily_evaluate_kernels(False)


class Evaluator:
    def __init__(
        self,
        dataset_base_folder,
        output_folder,
        use_full_dataset=True,
        fixed_kernel_mode=False,
        fixed_kernel_list=[BaseKernelTypes.SE],
    ):
        self.dataset_base_folder = dataset_base_folder
        self.output_folder = output_folder
        # self.data_set_list = [Energy, Concrete, Airfoil, AirlinePassenger, PowerPlant, Yacht, Wine, Boston, Kin8nm]
        self.data_set_list = [Energy, Concrete, Airfoil, AirlinePassenger, PowerPlant, Yacht, Wine]
        self.plot_data_set_names = [
            "Energy",
            "Concrete",
            "Airfoil",
            "Airline",
            "PowerPlant",
            "Yacht",
            "Wine",
            "Boston",
            "Kin8nm",
        ]
        self.use_full_dataset = use_full_dataset
        if self.use_full_dataset:
            # n_complete: Energy=768, Concrete=1030 Airfoil=1503 Airline=144 Powerplant=9568 Yacht=308 Wine=1599 Boston=506 Kin8nm=8192
            self.n_complete_list = [768, 1030, 1503, 144, 9568, 308, 1599, 506, 8192]
            fraction_train = 0.8
            self.n_data_list = [int(fraction_train * min(n_data, 2000)) for n_data in self.n_complete_list]
        else:
            self.n_data_list = [500, 500, 500, 100, 500, 250, 500, 400, 500]
        # for varying fraction of train dataset
        self.fraction_dataset = 1.0
        self.n_data_list = [int(n_data * self.fraction_dataset) for n_data in self.n_data_list]
        self.n_test = 400
        self.dataset_seed = 100
        self.use_absolute = True
        self.fraction = 0.8
        self.n_dataset_seeds_fixed_kernel_mode = 20
        dummy_kernel_config = BasicRBFPytorchConfig(input_dimension=0)

        ############## Amor configs ################
        self.n_iter_warm_start = 10
        self.lr_warm_start = 0.3
        self.gp_amor_tag = "GP-Amortized"
        self.gp_amor_warm_start_tag = "GP-Amortized warm-start"

        ############### Standard GP configuration ################
        self.gp_model_config = BasicGPModelPytorchConfig(kernel_config=dummy_kernel_config)
        self.gp_model_tag = "GP-ML"

        ############# Standard GP (only n_iter iterations) config #############
        self.gp_model_n_iter_config = BasicGPModelPytorchConfig(kernel_config=dummy_kernel_config)
        self.gp_model_n_iter_config.lr = self.lr_warm_start
        self.gp_model_n_iter_config.training_iter = self.n_iter_warm_start
        self.gp_model_n_iter_config.initial_likelihood_noise = 0.2
        self.gp_model_n_iter_tag = "GP-ML ({} iters)".format(self.n_iter_warm_start)

        ############ SVGP config #######################
        self.svgp_config = BasicSVGPModelPytorchConfig(kernel_config=dummy_kernel_config)
        self.svgp_model_tag = "SVGP"

        ############ SVGP (20 percent) config #######################
        self.svgp_20_config = BasicSVGPModelPytorchConfig(kernel_config=dummy_kernel_config)
        self.svgp_20_config.fraction_inducing_points = 0.2
        self.svgp_20_model_tag = "SVGP_20"

        ############ SVGP (50 percent) config #######################
        self.svgp_50_config = BasicSVGPModelPytorchConfig(kernel_config=dummy_kernel_config)
        self.svgp_50_config.fraction_inducing_points = 0.5
        self.svgp_50_model_tag = "SVGP_50"

        ############ GP Gpflow config #######################
        dummy_kernel_config_gpflow = BasicRBFConfig(input_dimension=0)
        self.gp_model_gpflow_config = GPModelFastConfig(kernel_config=dummy_kernel_config_gpflow)
        self.gp_model_gpflow_config.observation_noise = 0.2
        self.gp_model_gpflow_config.perturbation_for_singlestart_opt = 0.2
        self.gp_model_gpflow_config.set_prior_on_observation_noise = False
        self.gp_model_gpflow_tag = "GP-ML-gpflow"

        ############ Multi start GP configuration #####################
        self.gp_model_multistart_config = GPModelPytorchMultistartConfig(
            kernel_config=RBFWithPriorPytorchConfig(input_dimension=1)
        )
        self.gp_model_multistart_tag = "GP-ML (multi-start)"

        ############ Ensemble amor model config ##############
        self.gp_amor_ensemble_tag = "GP-Amortized ensemble"

        ############ Standard GP Ensemble config ############
        self.gp_ml_ensemble_tag = "GP-ML ensemble"
        self.gp_ml_ensemble_config = BasicGPModelPytorchEnsembleConfig()

        self.tags = [
            self.gp_amor_tag,
            self.gp_amor_ensemble_tag,
            self.gp_model_tag,
            self.gp_model_gpflow_tag,
            self.gp_amor_warm_start_tag,
            self.gp_model_n_iter_tag,
            self.svgp_model_tag,
            self.svgp_20_model_tag,
            self.svgp_50_model_tag,
            self.gp_model_multistart_tag,
            self.gp_ml_ensemble_tag,
        ]
        self.base_kernels = [
            BaseKernelTypes.SE,
            BaseKernelTypes.LIN,
            BaseKernelTypes.PER,
            BaseKernelTypes.LIN_MULT_PER,
            BaseKernelTypes.SE_MULT_LIN,
            BaseKernelTypes.SE_MULT_PER,
        ]
        self.model = None
        self.ensemble_model = None
        self.rmses_dict = {}
        self.nll_dict = {}
        self.infer_time_dict = {}
        self.time_dict = {}
        self.sampled_dataset_dict = {}
        self.success_dict = {}
        self.kernels = None
        self.fixed_kernel_list = fixed_kernel_list
        self.fixed_kernel_mode = fixed_kernel_mode
        if fixed_kernel_mode:
            self.sample_datasets_multiple_seeds()
            self.initialize_main_dict()
            self.kernels = [self.fixed_kernel_list]
        else:
            self.sample_datasets()
            self.initialize_main_dict()
            self.create_kernels()

    def set_model(self, model):
        self.model = model

    def load_model_from_checkpoint(self, checkpoint_path: str, model_config, load_to_ensemble: bool = False):
        self.model = GPModelAmortizedStructured(
            PredictionQuantity.PREDICT_Y, model_config, False, self.n_iter_warm_start, self.lr_warm_start
        )
        self.model.load_amortized_model(checkpoint_path, True)
        if load_to_ensemble:
            self.ensemble_model = GPModelAmortizedEnsemble(PredictionQuantity.PREDICT_Y, model_config)
            self.ensemble_model.load_amortized_model(checkpoint_path, True)

    def load_model_from_state_dict(self, state_dict, model_config, load_to_ensemble: bool = False):
        self.model = GPModelAmortizedStructured(
            PredictionQuantity.PREDICT_Y, model_config, False, self.n_iter_warm_start, self.lr_warm_start
        )
        self.model.build_model(state_dict)
        if load_to_ensemble:
            # @TODO implement this
            raise NotImplementedError

    def create_kernels(self):
        """
        Creates a list of kernel_lists -each kernel list e.g. [SE,PER] describes a kernel that is extended over all dimensions
        """
        kernels = [[kernel_type] for kernel_type in self.base_kernels]
        for i in range(2, 5):
            kernels_of_len = []
            while len(kernels_of_len) < 6:
                kernel_list_per_dim = list(np.random.choice(self.base_kernels, i, replace=False))
                kernel_list_per_dim.sort()
                if not self.check_if_in_kernels(kernel_list_per_dim, kernels):
                    kernels_of_len.append(kernel_list_per_dim)
            kernels += kernels_of_len
        for kernel_list in kernels:
            print(kernel_list)
        self.kernels = kernels

    def sample_datasets(self):
        for i, DatasetClass in enumerate(self.data_set_list):
            np.random.seed(self.dataset_seed)
            dataset_name = DatasetClass.__name__
            print(dataset_name)
            data_loader = DatasetClass(base_path=self.dataset_base_folder)
            assert isinstance(data_loader, BaseDataset)
            data_loader.load_data_set()
            self.n_data_list[i]
            x_data, y_data, x_test, y_test = data_loader.sample_train_test(
                use_absolute=self.use_absolute,
                n_train=self.n_data_list[i],
                n_test=self.n_test,
                fraction_train=self.fraction,
            )
            print("n_test: " + str(len(x_test)))
            self.sampled_dataset_dict[dataset_name] = (x_data, y_data, x_test, y_test)

    def sample_datasets_multiple_seeds(self):
        for i, DatasetClass in enumerate(self.data_set_list):
            np.random.seed(self.dataset_seed)
            dataset_name = DatasetClass.__name__
            self.sampled_dataset_dict[dataset_name] = []
            print(dataset_name)
            data_loader = DatasetClass(base_path=self.dataset_base_folder)
            assert isinstance(data_loader, BaseDataset)
            data_loader.load_data_set()
            self.n_data_list[i]
            for j in range(0, self.n_dataset_seeds_fixed_kernel_mode):
                x_data, y_data, x_test, y_test = data_loader.sample_train_test(
                    use_absolute=self.use_absolute,
                    n_train=self.n_data_list[i],
                    n_test=self.n_test,
                    fraction_train=self.fraction,
                )
                print("n_test: " + str(len(x_test)))
                self.sampled_dataset_dict[dataset_name].append((x_data, y_data, x_test, y_test))

    def initialize_main_dict(self):
        assert len(self.sampled_dataset_dict) > 0
        for dataset_name in self.sampled_dataset_dict:
            self.rmses_dict[dataset_name] = {}
            self.nll_dict[dataset_name] = {}
            self.time_dict[dataset_name] = {}
            self.infer_time_dict[dataset_name] = {}
            self.success_dict[dataset_name] = {}
            for tag in self.tags:
                self.rmses_dict[dataset_name][tag] = []
                self.nll_dict[dataset_name][tag] = []
                self.time_dict[dataset_name][tag] = []
                self.infer_time_dict[dataset_name][tag] = []
                self.success_dict[dataset_name][tag] = []

    def check_if_in_kernels(self, kernel_list, kernels):
        for kernel_list_in_kernels in kernels:
            if kernel_list == kernel_list_in_kernels:
                return True
        return False

    def eval_model_with_tag(self, tag):
        if self.fixed_kernel_mode:
            self.eval_model_with_tag_fixed_kernel(tag)
        else:
            self.eval_model_with_tag_standard(tag)

    def eval_model_with_tag_fixed_kernel(self, tag):
        for dataset_name in self.sampled_dataset_dict:
            n_seeds = len(self.sampled_dataset_dict[dataset_name])
            assert n_seeds == self.n_dataset_seeds_fixed_kernel_mode
            self.rmses_dict[dataset_name][tag] = []
            self.nll_dict[dataset_name][tag] = []
            self.time_dict[dataset_name][tag] = []
            self.infer_time_dict[dataset_name][tag] = []
            self.success_dict[dataset_name][tag] = []
            kernel_list = self.fixed_kernel_list
            for j in range(0, n_seeds):
                ###### Standard learning - amortized vs standard ##########
                success = False
                x_data, y_data, x_test, y_test = self.sampled_dataset_dict[dataset_name][j]
                n_dim = x_data.shape[1]
                try:
                    if tag == self.gp_amor_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_amortized_model(x_data, y_data, x_test, y_test, kernel_list)
                    elif tag == self.gp_amor_warm_start_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_model_amor_warm_start(x_data, y_data, x_test, y_test, kernel_list)
                    elif tag == self.gp_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_standard(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.gp_model_gpflow_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_gpflow(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.gp_model_n_iter_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_n_iter(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.svgp_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_svgp_model(x_data, y_data, x_test, y_test, n_dim, kernel_list, self.svgp_config)
                    elif tag == self.svgp_20_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_svgp_model(
                            x_data, y_data, x_test, y_test, n_dim, kernel_list, self.svgp_20_config
                        )
                    elif tag == self.svgp_50_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_svgp_model(
                            x_data, y_data, x_test, y_test, n_dim, kernel_list, self.svgp_50_config
                        )
                    elif tag == self.gp_model_multistart_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_multi_start(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.gp_ml_ensemble_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_ml_ensemble_model(x_data, y_data, x_test, y_test)
                    elif tag == self.gp_amor_ensemble_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_amortized_ensemble_model(x_data, y_data, x_test, y_test)
                    success = True
                except:
                    print("Error in eval for kernel list: " + str(kernel_list) + " and tag " + str(tag))
                if success:
                    self.rmses_dict[dataset_name][tag].append(rmse_model)
                    self.nll_dict[dataset_name][tag].append(nll_model)
                    self.time_dict[dataset_name][tag].append(time_after_model - time_before_model_infer)
                    self.infer_time_dict[dataset_name][tag].append(time_after_model_infer - time_before_model_infer)
                else:
                    self.rmses_dict[dataset_name][tag].append(None)
                    self.nll_dict[dataset_name][tag].append(None)
                    self.time_dict[dataset_name][tag].append(None)
                    self.infer_time_dict[dataset_name][tag].append(None)
                self.success_dict[dataset_name][tag].append(success)

    def eval_model_with_tag_standard(self, tag):
        for dataset_name in self.sampled_dataset_dict:
            x_data, y_data, x_test, y_test = self.sampled_dataset_dict[dataset_name]
            n_dim = x_data.shape[1]
            self.rmses_dict[dataset_name][tag] = []
            self.nll_dict[dataset_name][tag] = []
            self.time_dict[dataset_name][tag] = []
            self.infer_time_dict[dataset_name][tag] = []
            self.success_dict[dataset_name][tag] = []
            # ensemble models don't need to be evaluated over different kernels
            if tag == self.gp_ml_ensemble_tag:
                output_gp_ml_ensemble = self.eval_gp_ml_ensemble_model(x_data, y_data, x_test, y_test)
            elif tag == self.gp_amor_ensemble_tag:
                output_gp_amor_ensemble = self.eval_amortized_ensemble_model(x_data, y_data, x_test, y_test)

            for kernel_list in self.kernels:
                ###### Standard learning - amortized vs standard ##########
                success = False
                try:
                    if tag == self.gp_amor_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_amortized_model(x_data, y_data, x_test, y_test, kernel_list)
                    elif tag == self.gp_amor_warm_start_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_model_amor_warm_start(x_data, y_data, x_test, y_test, kernel_list)
                    elif tag == self.gp_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_standard(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.gp_model_gpflow_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_gpflow(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.gp_model_n_iter_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_n_iter(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.svgp_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_svgp_model(x_data, y_data, x_test, y_test, n_dim, kernel_list, self.svgp_config)
                    elif tag == self.svgp_20_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_svgp_model(
                            x_data, y_data, x_test, y_test, n_dim, kernel_list, self.svgp_20_config
                        )
                    elif tag == self.svgp_50_model_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_svgp_model(
                            x_data, y_data, x_test, y_test, n_dim, kernel_list, self.svgp_50_config
                        )
                    elif tag == self.gp_model_multistart_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = self.eval_gp_model_multi_start(x_data, y_data, x_test, y_test, n_dim, kernel_list)
                    elif tag == self.gp_ml_ensemble_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = output_gp_ml_ensemble
                    elif tag == self.gp_amor_ensemble_tag:
                        (
                            time_before_model_infer,
                            time_after_model_infer,
                            time_after_model,
                            rmse_model,
                            nll_model,
                        ) = output_gp_amor_ensemble
                    success = True
                except:
                    print("Error in eval for kernel list: " + str(kernel_list) + " and tag " + str(tag))
                if success:
                    self.rmses_dict[dataset_name][tag].append(rmse_model)
                    self.nll_dict[dataset_name][tag].append(nll_model)
                    self.time_dict[dataset_name][tag].append(time_after_model - time_before_model_infer)
                    self.infer_time_dict[dataset_name][tag].append(time_after_model_infer - time_before_model_infer)
                else:
                    self.rmses_dict[dataset_name][tag].append(None)
                    self.nll_dict[dataset_name][tag].append(None)
                    self.time_dict[dataset_name][tag].append(None)
                    self.infer_time_dict[dataset_name][tag].append(None)
                self.success_dict[dataset_name][tag].append(success)

    def save_dicts(self, prefix: str = ""):
        if self.fixed_kernel_mode:
            prefix = prefix + "_fixed_"
        pickle_object(self.rmses_dict, os.path.join(self.output_folder, f"{prefix}rmse_dict.pickle"))
        pickle_object(self.nll_dict, os.path.join(self.output_folder, f"{prefix}nll_dict.pickle"))
        pickle_object(self.infer_time_dict, os.path.join(self.output_folder, f"{prefix}infer_time_dict.pickle"))
        pickle_object(self.time_dict, os.path.join(self.output_folder, f"{prefix}time_dict.pickle"))
        pickle_object(self.success_dict, os.path.join(self.output_folder, f"{prefix}success_dict.pickle"))
        pickle_object(
            self.sampled_dataset_dict, os.path.join(self.output_folder, f"{prefix}sampled_dataset_dict.pickle")
        )
        pickle_object(self.kernels, os.path.join(self.output_folder, f"{prefix}kernel_list.pickle"))

    def rename_tag(self, tag, new_tag, delete_old=True):
        all_dicts = [self.rmses_dict, self.nll_dict, self.infer_time_dict, self.time_dict, self.success_dict]
        self.tags.append(new_tag)
        for single_dict in all_dicts:
            for dataset_name in self.sampled_dataset_dict:
                single_dict[dataset_name][new_tag] = single_dict[dataset_name][tag]
                if delete_old:
                    del single_dict[dataset_name][tag]

    def load_from_dicts(self, input_folder: str, prefix: str = "", check_datasets: bool = True):
        if self.fixed_kernel_mode:
            prefix = prefix + "_fixed_"
        rmses_dict = load_pickled_object(os.path.join(input_folder, f"{prefix}rmse_dict.pickle"))
        nll_dict = load_pickled_object(os.path.join(input_folder, f"{prefix}nll_dict.pickle"))
        print(nll_dict)
        infer_time_dict = load_pickled_object(os.path.join(input_folder, f"{prefix}infer_time_dict.pickle"))
        time_dict = load_pickled_object(os.path.join(input_folder, f"{prefix}time_dict.pickle"))
        success_dict = load_pickled_object(os.path.join(input_folder, f"{prefix}success_dict.pickle"))
        print(success_dict)
        sampled_dataset_dict = load_pickled_object(os.path.join(input_folder, f"{prefix}sampled_dataset_dict.pickle"))
        kernels = load_pickled_object(os.path.join(input_folder, f"{prefix}kernel_list.pickle"))
        self.load_subdict_to_dicts(rmses_dict, self.rmses_dict)
        self.load_subdict_to_dicts(nll_dict, self.nll_dict)
        self.load_subdict_to_dicts(infer_time_dict, self.infer_time_dict)
        self.load_subdict_to_dicts(success_dict, self.success_dict)
        self.load_subdict_to_dicts(time_dict, self.time_dict)
        if check_datasets:
            if self.fixed_kernel_mode:
                for dataset_name in sampled_dataset_dict:
                    if dataset_name in self.sampled_dataset_dict:
                        for j in range(0, self.n_dataset_seeds_fixed_kernel_mode):
                            x_data, y_data, x_test, y_test = self.sampled_dataset_dict[dataset_name][j]
                            x_data_load, y_data_load, x_test_load, y_test_load = sampled_dataset_dict[dataset_name][j]
                            assert np.allclose(x_data, x_data_load)
                            assert np.allclose(y_data, y_data_load)
                            assert np.allclose(x_test, x_test_load)
                            assert np.allclose(y_test, y_test_load)
            else:
                for dataset_name in sampled_dataset_dict:
                    if dataset_name in self.sampled_dataset_dict:
                        x_data, y_data, x_test, y_test = self.sampled_dataset_dict[dataset_name]
                        x_data_load, y_data_load, x_test_load, y_test_load = sampled_dataset_dict[dataset_name]
                        assert np.allclose(x_data, x_data_load)
                        assert np.allclose(y_data, y_data_load)
                        assert np.allclose(x_test, x_test_load)
                        assert np.allclose(y_test, y_test_load)
            assert self.kernels == kernels

    def load_subdict_to_dicts(self, subdict: dict, main_dict: dict):
        for dataset_name in subdict:
            if dataset_name in main_dict:
                for tag in subdict[dataset_name]:
                    if tag in main_dict[dataset_name] and len(subdict[dataset_name][tag]) > 0:
                        main_dict[dataset_name][tag] = subdict[dataset_name][tag]

    def create_plots(
        self,
        return_fig: bool = False,
        save_fig: bool = False,
        bar_plot: bool = False,
        log_y_time: bool = False,
        restrict_to_tags: bool = False,
        use_tags_list=[],
        restrict_to_dataset: bool = False,
        dataset_name_list=[],
        normalize_time: bool = False,
        return_nll: bool = False,
        return_time: bool = False,
    ):
        rmse_dict, nll_dict, time_dict, infer_time_dict = self.create_pruned_dicts(
            restrict_to_tags, use_tags_list, restrict_to_dataset, dataset_name_list
        )
        rmse_dict = self.change_dataset_names(rmse_dict)
        nll_dict = self.change_dataset_names(nll_dict)
        time_dict = self.change_dataset_names(time_dict)
        if normalize_time:
            time_label = "Time-Ratio"
            time_dict = self.divide_by_median_of_tag(time_dict, self.gp_amor_tag)
        else:
            time_label = "Time (sec)"
        if return_fig:
            if return_time:
                fig_time = self.create_boxplot(
                    time_dict,
                    time_label,
                    "TIME_box_plots.png",
                    return_fig=return_fig,
                    save_fig=save_fig,
                    bar_plot=bar_plot,
                    log_y=log_y_time,
                )
                return None, None, fig_time
            elif return_nll:
                fig_nll = self.create_boxplot(
                    nll_dict,
                    "NLL",
                    "NLL_box_plots.png",
                    return_fig=return_fig,
                    save_fig=save_fig,
                    bar_plot=bar_plot,
                    log_y=False,
                )
                return None, fig_nll, None
            else:
                fig_rmse = self.create_boxplot(
                    rmse_dict,
                    "RMSE",
                    "RMSE_box_plots.png",
                    return_fig=return_fig,
                    save_fig=save_fig,
                    bar_plot=bar_plot,
                    log_y=False,
                )
                return fig_rmse, None, None
        else:
            fig_rmse = self.create_boxplot(
                rmse_dict,
                "RMSE",
                "RMSE_box_plots.png",
                return_fig=return_fig,
                save_fig=save_fig,
                bar_plot=bar_plot,
                log_y=False,
            )
            fig_nll = self.create_boxplot(
                nll_dict,
                "NLL",
                "NLL_box_plots.png",
                return_fig=return_fig,
                save_fig=save_fig,
                bar_plot=bar_plot,
                log_y=False,
            )
            fig_time = self.create_boxplot(
                time_dict,
                time_label,
                "TIME_box_plots.png",
                return_fig=return_fig,
                save_fig=save_fig,
                bar_plot=bar_plot,
                log_y=log_y_time,
            )
        return None, None, None

    def get_cleaned_dicts(
        self, restrict_to_tags: bool = False, use_tags_list=[], restrict_to_dataset: bool = False, dataset_name_list=[]
    ):
        rmse_dict, nll_dict, time_dict, infer_time_dict = self.create_pruned_dicts(
            restrict_to_tags, use_tags_list, restrict_to_dataset, dataset_name_list
        )
        rmse_dict = self.change_dataset_names(rmse_dict)
        nll_dict = self.change_dataset_names(nll_dict)
        time_dict = self.change_dataset_names(time_dict)
        time_dict = self.divide_by_median_of_tag(time_dict, self.gp_amor_tag)
        return rmse_dict, nll_dict, time_dict

    def get_statistics(
        self,
        restrict_to_tags: bool = False,
        use_tags_list=[],
        restrict_to_dataset: bool = False,
        dataset_name_list=[],
        divide_time: bool = True,
    ):
        rmse_dict, nll_dict, time_dict, infer_time_dict = self.create_pruned_dicts(
            restrict_to_tags, use_tags_list, restrict_to_dataset, dataset_name_list
        )
        rmse_stat_dict = self.extract_statistics_from_dict(self.change_dataset_names(rmse_dict))
        nll_stat_dict = self.extract_statistics_from_dict(self.change_dataset_names(nll_dict))
        time_dict = self.change_dataset_names(time_dict)
        if divide_time:
            time_dict = self.divide_by_median_of_tag(time_dict, self.gp_amor_tag)
        time_stat_dict = self.extract_statistics_from_dict(time_dict)
        return rmse_stat_dict, nll_stat_dict, time_stat_dict

    def extract_statistics_from_dict(self, metric_dist):
        statistics_dict = {}
        for dataset_name in metric_dist:
            statistics_dict[dataset_name] = {}
            for tag in metric_dist[dataset_name]:
                statistics_dict[dataset_name][tag] = {}
                metrics_array = metric_dist[dataset_name][tag]
                statistics_dict[dataset_name][tag]["median"] = np.median(metrics_array)
                statistics_dict[dataset_name][tag]["q20"] = np.quantile(metrics_array, 0.2)
                statistics_dict[dataset_name][tag]["q25"] = np.quantile(metrics_array, 0.25)
                statistics_dict[dataset_name][tag]["q75"] = np.quantile(metrics_array, 0.75)
                statistics_dict[dataset_name][tag]["q80"] = np.quantile(metrics_array, 0.8)
                statistics_dict[dataset_name][tag]["std"] = np.std(metrics_array)
                statistics_dict[dataset_name][tag]["mean"] = np.mean(metrics_array)
        return statistics_dict

    def divide_by_median_of_tag(self, input_dict, target_tag):
        new_dict = {}
        for dataset_name in input_dict:
            new_dict[dataset_name] = {}
            print(target_tag)
            median_target_tag = np.median(input_dict[dataset_name][target_tag])
            for tag in input_dict[dataset_name]:
                new_dict[dataset_name][tag] = np.array(input_dict[dataset_name][tag]) / median_target_tag
        return new_dict

    def change_dataset_names(self, input_dict):
        new_dict = {}
        for i, DatasetClass in enumerate(self.data_set_list):
            dataset_name = DatasetClass.__name__
            new_dataset_name = self.plot_data_set_names[i]
            if dataset_name in input_dict:
                new_dict[new_dataset_name] = input_dict[dataset_name]
        return new_dict

    def create_pruned_dicts(
        self, restrict_to_tags: bool = False, use_tags_list=[], restrict_to_dataset: bool = False, dataset_name_list=[]
    ):
        new_rmse_dict = {}
        new_nll_dict = {}
        new_time_dict = {}
        new_infer_time_dict = {}
        if restrict_to_dataset:
            dataset_names = dataset_name_list
        else:
            dataset_names = list(self.rmses_dict.keys())
        for dataset_name in dataset_names:
            if restrict_to_tags:
                tags = [
                    tag
                    for tag in use_tags_list
                    if tag in self.rmses_dict[dataset_name].keys() and len(self.rmses_dict[dataset_name][tag]) > 0
                ]
            else:
                tags = [
                    tag for tag in self.rmses_dict[dataset_name].keys() if len(self.rmses_dict[dataset_name][tag]) > 0
                ]
            # tags = list(self.rmses_dict[dataset_name].keys())
            key_0 = tags[0]
            n = len(self.success_dict[dataset_name][key_0])
            new_rmse_dict[dataset_name] = {}
            new_nll_dict[dataset_name] = {}
            new_time_dict[dataset_name] = {}
            new_infer_time_dict[dataset_name] = {}
            for tag in tags:
                new_rmse_dict[dataset_name][tag] = []
                new_nll_dict[dataset_name][tag] = []
                new_time_dict[dataset_name][tag] = []
                new_infer_time_dict[dataset_name][tag] = []

            for i in range(0, n):
                all_succeeded = True
                for tag in tags:
                    if not self.success_dict[dataset_name][tag][i]:
                        all_succeeded = False
                if all_succeeded:
                    for tag in tags:
                        new_rmse_dict[dataset_name][tag].append(self.rmses_dict[dataset_name][tag][i])
                        new_nll_dict[dataset_name][tag].append(self.nll_dict[dataset_name][tag][i])
                        new_time_dict[dataset_name][tag].append(self.time_dict[dataset_name][tag][i])
                        new_infer_time_dict[dataset_name][tag].append(self.infer_time_dict[dataset_name][tag][i])
        return new_rmse_dict, new_nll_dict, new_time_dict, new_infer_time_dict

    def eval_amortized_model(self, x_data, y_data, x_test, y_test, kernel_list):
        self.model.set_do_warm_start(False)

        self.model.clear_cache()

        self.model.set_kernel_list(kernel_list)

        time_before_model_infer = time.perf_counter()

        self.model.infer(x_data, y_data)

        time_after_model_infer = time.perf_counter()

        pred_mu_model, pred_sigma_model = self.model.predictive_dist(x_test)

        time_after_model = time.perf_counter()

        rmse_model = calculate_rmse(pred_mu_model, y_test)

        nll_model = np.mean(-1 * self.model.predictive_log_likelihood(x_test, y_test))
        return time_before_model_infer, time_after_model_infer, time_after_model, rmse_model, nll_model

    def eval_amortized_ensemble_model(self, x_data, y_data, x_test, y_test):
        self.ensemble_model.fast_batch_inference = True

        self.ensemble_model.set_kernel_list(self.kernels)

        time_before_model_infer = time.perf_counter()

        self.ensemble_model.infer(x_data, y_data)

        time_after_model_infer = time.perf_counter()

        pred_mu_model, pred_sigma_model = self.ensemble_model.predictive_dist(x_test)

        time_after_model = time.perf_counter()

        rmse_model = calculate_rmse(pred_mu_model, y_test)

        nll_model = np.mean(-1 * self.ensemble_model.predictive_log_likelihood(x_test, y_test))
        return time_before_model_infer, time_after_model_infer, time_after_model, rmse_model, nll_model

    def eval_model_amor_warm_start(self, x_data, y_data, x_test, y_test, kernel_list):
        self.model.set_do_warm_start(True)

        self.model.clear_cache()

        self.model.set_kernel_list(kernel_list)

        time_before_model_warm_infer = time.perf_counter()

        self.model.infer(x_data, y_data)

        time_after_model_warm_infer = time.perf_counter()

        pred_mu_model_warm, pred_sigma_model_warm = self.model.predictive_dist(x_test)

        time_after_model_warm = time.perf_counter()

        rmse_model_warm = calculate_rmse(pred_mu_model_warm, y_test)

        nll_model_warm = np.mean(-1 * self.model.predictive_log_likelihood(x_test, y_test))
        return (
            time_before_model_warm_infer,
            time_after_model_warm_infer,
            time_after_model_warm,
            rmse_model_warm,
            nll_model_warm,
        )

    def eval_gp_model_n_iter(self, x_data, y_data, x_test, y_test, n_dim, kernel_list):
        full_kernel_list = self.model.create_input_kernel_list(kernel_list, n_dim)

        kernel_expression = transform_kernel_list_to_expression(full_kernel_list[0], add_prior=False)

        gp_model_n_iter = ModelFactory.build(self.gp_model_n_iter_config)
        gp_model_n_iter.kernel_module = kernel_expression.get_kernel()

        time_before_gp_model_n_iter_infer = time.perf_counter()

        gp_model_n_iter.infer(x_data, y_data)

        time_after_gp_model_n_iter_infer = time.perf_counter()

        gp_pred_mu_n_iter, gp_pred_sigma_n_iter = gp_model_n_iter.predictive_dist(x_test)

        time_after_gp_model_n_iter = time.perf_counter()

        gp_rmse_n_iter = calculate_rmse(gp_pred_mu_n_iter, y_test)

        gp_nll_n_iter = np.mean(-1 * gp_model_n_iter.predictive_log_likelihood(x_test, y_test))
        return (
            time_before_gp_model_n_iter_infer,
            time_after_gp_model_n_iter_infer,
            time_after_gp_model_n_iter,
            gp_rmse_n_iter,
            gp_nll_n_iter,
        )

    def eval_gp_model_standard(self, x_data, y_data, x_test, y_test, n_dim, kernel_list):
        full_kernel_list = self.model.create_input_kernel_list(kernel_list, n_dim)

        kernel_expression = transform_kernel_list_to_expression(full_kernel_list[0], add_prior=False)

        gp_model = ModelFactory.build(self.gp_model_config)
        gp_model.kernel_module = kernel_expression.get_kernel()

        time_before_gp_model_infer = time.perf_counter()

        gp_model.infer(x_data, y_data)

        time_after_gp_model_infer = time.perf_counter()

        gp_pred_mu, gp_pred_sigma = gp_model.predictive_dist(x_test)

        time_after_gp_model = time.perf_counter()

        gp_rmse = calculate_rmse(gp_pred_mu, y_test)

        gp_nll = np.mean(-1 * gp_model.predictive_log_likelihood(x_test, y_test))
        return time_before_gp_model_infer, time_after_gp_model_infer, time_after_gp_model, gp_rmse, gp_nll

    def eval_gp_model_gpflow(self, x_data, y_data, x_test, y_test, n_dim, kernel_list):
        full_kernel_list = self.model.create_input_kernel_list(kernel_list, n_dim)

        kernel_expression = transform_kernel_list_to_expression(full_kernel_list[0], add_prior=False, use_gpflow=True)

        gp_model = ModelFactory.build(self.gp_model_gpflow_config)
        gp_model.set_kernel(kernel_expression.get_kernel())

        time_before_gp_model_infer = time.perf_counter()

        gp_model.infer(x_data, y_data)

        time_after_gp_model_infer = time_before_gp_model_infer + gp_model.get_last_inference_time()

        time_before_gp_model_pred = time.perf_counter()

        gp_pred_mu, gp_pred_sigma = gp_model.predictive_dist(x_test)

        time_after_gp_model_pred = time.perf_counter()

        pred_time = time_after_gp_model_pred - time_before_gp_model_pred

        time_after_gp_model = time_after_gp_model_infer + pred_time

        gp_rmse = calculate_rmse(gp_pred_mu, y_test)

        gp_nll = np.mean(-1 * gp_model.predictive_log_likelihood(x_test, y_test))
        return time_before_gp_model_infer, time_after_gp_model_infer, time_after_gp_model, gp_rmse, gp_nll

    def eval_gp_ml_ensemble_model(self, x_data, y_data, x_test, y_test):
        gp_ml_ensemble = ModelFactory.build(self.gp_ml_ensemble_config)
        gp_ml_ensemble.set_kernel_list(self.kernels)

        time_before_model_infer = time.perf_counter()

        gp_ml_ensemble.infer(x_data, y_data)

        time_after_model_infer = time.perf_counter()

        pred_mu_model, pred_sigma_model = gp_ml_ensemble.predictive_dist(x_test)

        time_after_model = time.perf_counter()

        rmse_model = calculate_rmse(pred_mu_model, y_test)

        nll_model = np.mean(-1 * gp_ml_ensemble.predictive_log_likelihood(x_test, y_test))
        return time_before_model_infer, time_after_model_infer, time_after_model, rmse_model, nll_model

    def eval_gp_model_multi_start(self, x_data, y_data, x_test, y_test, n_dim, kernel_list):
        full_kernel_list = self.model.create_input_kernel_list(kernel_list, n_dim)

        kernel_expression = transform_kernel_list_to_expression(full_kernel_list[0], add_prior=True)

        gp_model = ModelFactory.build(self.gp_model_multistart_config)
        gp_model.kernel_module = gpytorch.kernels.AdditiveKernel(kernel_expression.get_kernel())

        time_before_gp_model_infer = time.perf_counter()

        gp_model.infer(x_data, y_data)

        time_after_gp_model_infer = time.perf_counter()

        gp_pred_mu, gp_pred_sigma = gp_model.predictive_dist(x_test)

        time_after_gp_model = time.perf_counter()

        gp_rmse = calculate_rmse(gp_pred_mu, y_test)

        gp_nll = np.mean(-1 * gp_model.predictive_log_likelihood(x_test, y_test))
        return time_before_gp_model_infer, time_after_gp_model_infer, time_after_gp_model, gp_rmse, gp_nll

    def eval_svgp_model(self, x_data, y_data, x_test, y_test, n_dim, kernel_list, svgp_config):
        full_kernel_list = self.model.create_input_kernel_list(kernel_list, n_dim)

        kernel_expression = transform_kernel_list_to_expression(full_kernel_list[0], add_prior=False)

        gp_model = ModelFactory.build(svgp_config)
        gp_model.kernel_module = kernel_expression.get_kernel()

        time_before_gp_model_infer = time.perf_counter()

        gp_model.infer(x_data, y_data)

        time_after_gp_model_infer = time.perf_counter()

        gp_pred_mu, gp_pred_sigma = gp_model.predictive_dist(x_test)

        time_after_gp_model = time.perf_counter()

        gp_rmse = calculate_rmse(gp_pred_mu, y_test)

        gp_nll = np.mean(-1 * gp_model.predictive_log_likelihood(x_test, y_test))
        return time_before_gp_model_infer, time_after_gp_model_infer, time_after_gp_model, gp_rmse, gp_nll

    def create_boxplot(
        self,
        dictionary,
        y_name,
        file_name,
        x_axis_is_dataset: bool = True,
        return_fig: bool = False,
        save_fig: bool = False,
        bar_plot: bool = True,
        log_y: bool = False,
    ):
        fig = create_box_plot_from_dict(
            dictionary,
            y_name,
            "Dataset",
            "Method",
            not x_axis_is_dataset,
            save_fig,
            self.output_folder,
            file_name,
            return_fig,
            bar_plot=bar_plot,
            log_y=log_y,
        )
        return fig
