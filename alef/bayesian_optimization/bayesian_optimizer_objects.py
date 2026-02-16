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

from typing import List, Union
import numpy as np
from numpy import random
from alef.acquisition_function.al_acquisition_functions.acq_random import Random
from alef.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
from alef.bayesian_optimization.bayesian_optimizer import BayesianOptimizer
from alef.bayesian_optimization.duration_time_predictors_objects import BaseDurationTimePredictorObjects
from alef.enums.bayesian_optimization_enums import (
    AcquisitionOptimizationObjectBOType,
    ValidationType,
)
from alef.bayesian_optimization.evolutionary_optimizer_objects import EvolutionaryOptimizerObjects
from alef.kernels.kernel_kernel_grammar_tree import OptimalTransportKernelKernel
from alef.oracles.base_object_oracle import BaseObjectOracle
from alef.models.object_gp_model import ObjectGpModel
import logging
from gpflow.utilities.traversal import tabulate_module_summary
from gpflow.config import default_summary_fmt
import time
from alef.oracles.gp_model_bic_oracle import GPModelBICOracle
from alef.oracles.gp_model_cv_oracle import GPModelCVOracle
from alef.oracles.gp_model_evidence_oracle import GPModelEvidenceOracle
from alef.bayesian_optimization.base_candidate_generator import CandidateGenerator

logger = logging.getLogger(__name__)


class BayesianOptimizerObjects(BayesianOptimizer):
    def __init__(
        self,
        acquisition_function: Union[BaseBOAcquisitionFunction, Random],
        validation_type: ValidationType,
        acquisiton_optimization_type: AcquisitionOptimizationObjectBOType,
        population_evolutionary: int,
        n_steps_evolutionary: int,
        num_offspring_evolutionary: int,
        n_prune_trailing: int,
        do_plotting: bool = False,
        use_duration_time_predictor_in_acquisition: bool = False,
        **kwargs,
    ):
        super().__init__(acquisition_function, validation_type, None, do_plotting, None, None)
        self.acquisiton_optimization_type = acquisiton_optimization_type
        self.current_bests = []
        self.query_list = []
        self.additional_metrics = []
        self.iteration_time_list = []
        self.acquisition_time_list = []
        self.oracle_time_list = []
        self.query_durations_x_data = []
        self.kernel_parameters_list = []
        self.additional_metrics_index_interval = 10
        self.n_prune_trailing = n_prune_trailing
        self.population_evolutionary = population_evolutionary
        self.n_steps_evolutionary = n_steps_evolutionary
        self.num_offspring_evolutionary = num_offspring_evolutionary
        self.use_duration_time_predictor_in_acquisition = use_duration_time_predictor_in_acquisition

    def set_oracle(self, oracle: BaseObjectOracle):
        """
        Set the oracle.
        """
        self.oracle = oracle

    def set_model(self, model: ObjectGpModel):
        """
        Set the model.
        """
        self.model = model

    def set_candidate_generator(self, generator: CandidateGenerator):
        """
        Set the candidate generator.
        """
        self.candidate_generator = generator

    def set_duration_time_predictor(self, predictor: BaseDurationTimePredictorObjects):
        """
        Set the duration time predictor.
        """
        self.duration_time_predictor = predictor

    def set_evolutionary_opt_settings(self, population: int, n_steps: int, num_offspring: int):
        """
        Set the settings for the evolutionary optimizer.
        """
        self.population_evolutionary = population
        self.n_steps_evolutionary = n_steps
        self.num_offspring_evolutionary = num_offspring

    def sample_ground_truth(self):
        """
        Sample the ground truth data.
        """
        raise NotImplementedError

    def sample_from_oracle_to_find_max_value(self, n_data: int):
        """
        Sample data from the oracle to find the maximum value.
        """
        raise NotImplementedError

    def set_max_value_for_validation(self, value: float):
        """
        Set the maximum value for validation.
        """
        return super().set_max_value_for_validation(value)

    def sample_train_set(self, n_data, seed=100, set_seed=False):
        """
        Sample the training set.
        """
        self.x_data = self.candidate_generator.get_random_canditates(n_data, seed, set_seed)
        y_list = []
        logger.info("Sample train set")
        for x in self.x_data:
            logger.info("Sample: " + str(x))
            y, query_duration = self.oracle.query(x)
            logger.info("Output: " + str(y))
            y_list.append(y)
            self.query_durations_x_data.append(query_duration)
        self.y_data = np.expand_dims(np.array(y_list), axis=1)

    def set_train_set(self, x_train: List[object], y_train: np.array, query_durations_x_train: List[float]):
        """
        Set the training set.
        """
        self.x_data = x_train
        self.y_data = y_train
        self.query_durations_x_data = query_durations_x_train
        assert len(self.query_durations_x_data) == len(self.x_data)

    def update(self, step: int):
        """
        Update the optimizer.
        """
        if isinstance(self.acquisition_function, Random):
            query = random.choice(self.candidates)
            return query, None

        if self.use_duration_time_predictor_in_acquisition:
            self.duration_time_predictor.fit(self.x_data, self.query_durations_x_data)

        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        self.log_kernel_parameters()
        logger.info(tabulate_module_summary(self.model.model, default_summary_fmt()))
        if self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES:
            # if self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES
            acquisition_values = self.acquisition_function_wrapper(self.candidates)
            new_query = self.candidates[np.argmax(acquisition_values)]
            logger.info("Current best:")
            logger.info(self.get_current_best())
            return new_query, acquisition_values

        elif self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.EVOLUTIONARY:
            optimizer = EvolutionaryOptimizerObjects(
                self.population_evolutionary, self.num_offspring_evolutionary, self.candidate_generator
            )
            new_query, _ = optimizer.maximize(self.acquisition_function_wrapper, self.n_steps_evolutionary)
            return new_query, None

    def acquisition_function_wrapper(self, x_grid: List[object]) -> np.array:
        """
        Wrapper for the acquisition function.
        """
        assert isinstance(self.acquisition_function, BaseBOAcquisitionFunction)
        score = self.acquisition_function.acquisition_score(x_grid, self.model, self.x_data, self.y_data)
        if self.use_duration_time_predictor_in_acquisition:
            duration_predictions = self.duration_time_predictor.predict(x_grid)
            assert len(score) == len(duration_predictions)
            score = score / duration_predictions
        return score

    def maximize(self, n_steps: int):
        """
        Maximize the objective function.
        """
        self.n_steps = n_steps
        if self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES or isinstance(
            self.acquisition_function, Random
        ):
            self.candidates = self.candidate_generator.get_initial_candidates_trailing()
        self.add_additional_metrics(0)
        self.validate()
        for i in range(0, self.n_steps):
            time_before_iteration = time.perf_counter()
            time_before_acquisition = time.perf_counter()
            query, acquisition_values_candidates = self.update(i)
            time_after_acquisition = time.perf_counter()
            logger.info("Query:")
            logger.info(query)
            time_before_oracle = time.perf_counter()
            new_y, query_duration = self.oracle.query(query)
            time_after_oracle = time.perf_counter()
            logger.info("Oracle output:")
            logger.info(new_y)
            self.current_bests.append((self.get_current_best(), self.get_current_best_value()))
            self.query_list.append((query, np.float(new_y)))
            self.x_data.append(query)
            self.y_data = np.vstack((self.y_data, [new_y]))
            self.query_durations_x_data.append(query_duration)
            assert len(self.x_data) == len(self.query_durations_x_data)
            self.add_additional_metrics(i + 1)
            self.validate()
            if self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES:
                self.update_candidates(acquisition_values_candidates)
            if self.do_plotting:
                self.plot_validation_curve()
            time_after_iteration = time.perf_counter()
            time_diff_acquisition = time_after_acquisition - time_before_acquisition
            time_diff_oracle = time_after_oracle - time_before_oracle
            time_diff_iteration = time_after_iteration - time_before_iteration
            self.iteration_time_list.append(time_diff_iteration)
            self.acquisition_time_list.append(time_diff_acquisition)
            self.oracle_time_list.append(time_diff_oracle)
        return (
            np.array(self.validation_metrics),
            self.query_list,
            self.current_bests,
            self.additional_metrics,
            self.iteration_time_list,
            self.oracle_time_list,
            self.acquisition_time_list,
            self.kernel_parameters_list,
        )

    def update_candidates(self, acquisition_values):
        """
        Update the candidates.
        """
        # First prune candidates to throw away candidates with low acquisition values and ones inside dataset
        self.candidates = self.get_pruned_candidates(acquisition_values)
        # Increase candidates with new random candidates and new candidates around the current best one
        additional_candidates = []
        generator_candidates = self.candidate_generator.get_additional_candidates_trailing(self.get_current_best())
        for candidate in generator_candidates:
            if not self.check_if_in_list(candidate, self.candidates) and not self.check_if_in_list(
                candidate, self.x_data
            ):
                additional_candidates.append(candidate)
        self.candidates = self.candidates + additional_candidates

    def get_pruned_candidates(self, acquisition_values):
        """
        Get the pruned candidates.
        """
        pruned_candidates = []
        best_indexes = np.argsort(-1 * acquisition_values)[: self.n_prune_trailing]
        assert acquisition_values[best_indexes[0]] >= acquisition_values[best_indexes[1]]
        for index in best_indexes:
            candidate = self.candidates[index]
            if not self.check_if_in_list(candidate, pruned_candidates) and not self.check_if_in_list(
                candidate, self.x_data
            ):
                pruned_candidates.append(self.candidates[index])
        return pruned_candidates

    def check_if_in_list(self, object_element, object_list):
        """
        Check if an object is in a list.
        """
        for object_list_element in object_list:
            if str(object_element) == str(object_list_element):
                return True
        return False

    def get_current_best(self):
        """
        Get the current best candidate.
        """
        return self.x_data[np.argmax(self.y_data)]

    def get_current_best_value(self):
        """
        Get the current best value.
        """
        return np.max(self.y_data)

    def validate(self):
        """
        Validate the model.
        """
        return super().validate()

    def add_additional_metrics(self, index):
        """
        Add additional metrics.
        """
        if index % self.additional_metrics_index_interval == 0 or index == self.n_steps:
            if (
                isinstance(self.oracle, GPModelBICOracle)
                or isinstance(self.oracle, GPModelEvidenceOracle)
                or isinstance(self.oracle, GPModelCVOracle)
            ):
                if self.oracle.x_test is not None:
                    additional_metrics = self.oracle.query_on_test_set(self.get_current_best())
                    self.additional_metrics.append((index, *additional_metrics))

    def log_kernel_parameters(self):
        """
        Log the kernel parameters.
        """
        if isinstance(self.model.model.kernel, OptimalTransportKernelKernel):
            alphas = self.model.model.kernel.alphas.numpy()
            lengthscale = self.model.model.kernel.lengthscale.numpy()
            variance = self.model.model.kernel.variance.numpy()
            likelihood_variance = self.model.model.likelihood.variance.numpy()
            parameter_vector = np.concatenate((alphas, [lengthscale], [variance], [likelihood_variance]))
            self.kernel_parameters_list.append(parameter_vector)
