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

from enum import Enum
from typing import Optional, Tuple
import numpy as np
from alef.bayesian_optimization.bayesian_optimizer_factory import BayesianOptimizerFactory
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.models.gp_model_config import GPModelExtenseOptimization
from alef.configs.models.object_gp_model_config import BasicObjectGPModelConfig
from alef.kernels.kernel_factory import KernelFactory
from alef.kernels.kernel_grammar.kernel_grammar_candidate_generator import KernelGrammarCandidateGenerator
from alef.models.base_model import BaseModel
from alef.models.gp_model import GPModel
from alef.models.object_mean_functions import ObjectConstant
from alef.oracles.gp_model_bic_oracle import GPModelBICOracle
from alef.oracles.gp_model_evidence_oracle import GPModelEvidenceOracle
from alef.oracles.gp_model_cv_oracle import GPModelCVOracle
from alef.configs.bayesian_optimization.bayesian_optimizer_objects_configs import ObjectBOExpectedImprovementEAConfig
from alef.configs.kernels.grammar_tree_kernel_kernel_configs import (
    OTWeightedDimsExtendedGrammarKernelConfig,
    OTWeightedDimsExtendedKernelWithHyperpriorConfig,
)
from alef.models.object_gp_model import ObjectGpModel, PredictionQuantity as PredictionQuantityMetaModel
import logging

logger = logging.getLogger(__name__)


class OracleType(Enum):
    BIC = 0
    EVIDENCE = 1
    CV = 2


class GPModelKernelSearch(BaseModel):
    """
    Model that internally makes a kernel search and uses the best kernel found for predictions. Kernel search is conducted via BO over the kernel grammar with a
    specified model selection criteria: Log-Evidence, BIC or cross-validation Log-Likelihood.

    Attributes:
        grammar_generator: KernelGrammarCandidateGenerator object that is used by the BO over kernel space and which defines the kernel search space
        oracle_type: OracleType enum defining which model selection criteria should be used for kernel serch
        fast_inference: boolean that specifies if the model selection criteria should be evaluted fast - this influences mainly how many restarts are done for MAP estimates of the kernel hps
        n_steps_bo: int specifying how many kernel evaluations should be done in the BO procedure
        kernel_kernel_config : config specfiying which kernel-kernel should be used for BO over kernel space
    """

    def __init__(
        self,
        grammar_generator: KernelGrammarCandidateGenerator,
        oracle_type: OracleType,
        fast_inference: bool,
        n_steps_bo: int,
        use_meta_gp_hp_prior: bool,
        default_n_steps_ea: int,
        population_size_tuple: Tuple[int, int],
        bo_n_initial_factor: int,
        **kwargs,
    ) -> None:
        self.grammar_generator = grammar_generator
        self.oracle_type = oracle_type
        self.fast_inference = fast_inference
        self.default_n_steps_ea = default_n_steps_ea
        self.n_steps_bo = n_steps_bo
        self.population_size_fast_inference = population_size_tuple[0]
        self.population_size_standard = population_size_tuple[1]
        self.input_dimension = self.grammar_generator.get_input_dimension()
        self.bo_config = self.get_bo_config()
        self.n_initial_bo = bo_n_initial_factor * self.grammar_generator.search_space.get_num_base_kernels()
        if use_meta_gp_hp_prior:
            self.kernel_kernel_config = OTWeightedDimsExtendedKernelWithHyperpriorConfig()
        else:
            self.kernel_kernel_config = OTWeightedDimsExtendedGrammarKernelConfig()
        self.object_gp_model_config = BasicObjectGPModelConfig(
            kernel_config=self.kernel_kernel_config,
            prediction_quantity=PredictionQuantityMetaModel.PREDICT_F,
            perform_multi_start_optimization=False,
            set_prior_on_observation_noise=use_meta_gp_hp_prior,
        )
        self.final_model_config = GPModelExtenseOptimization(
            kernel_config=BaseKernelConfig(input_dimension=0, name="dummy")
        )
        self.model = None

    def get_bo_config(self):
        """
        Build BO config depending if fast_inference is activated and depending of the search space size - mainly specifies the acquisition optimization parameters
        """
        if self.grammar_generator.search_space.considers_single_dimensions():
            n_steps_ea = max(self.input_dimension, self.default_n_steps_ea)
        else:
            n_steps_ea = self.default_n_steps_ea

        if self.fast_inference:
            population_size = self.population_size_fast_inference
        else:
            population_size = self.population_size_standard
        return ObjectBOExpectedImprovementEAConfig(
            n_steps_evolutionary=n_steps_ea, population_evolutionary=population_size
        )

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Main search loop - gets input data x_data,y_data for which a model selection criteria can be calculated
        BO over kernel space is used to find the kernel for which the GP has highest model selection criteria value
        for the given dataset - found best kernel is stored in the GPModel in self.model
        """
        if self.oracle_type == OracleType.BIC:
            self.oracle = GPModelBICOracle(x_data, y_data, self.grammar_generator, self.fast_inference)
        elif self.oracle_type == OracleType.EVIDENCE:
            self.oracle = GPModelEvidenceOracle(x_data, y_data, self.grammar_generator, self.fast_inference)
        elif self.oracle_type == OracleType.CV:
            self.oracle = GPModelCVOracle(x_data, y_data, self.grammar_generator, self.fast_inference)
        assert x_data.shape[1] == self.input_dimension
        optimizer = BayesianOptimizerFactory.build(self.bo_config)
        kernel_kernel = KernelFactory.build(self.kernel_kernel_config)
        object_gp = ObjectGpModel(kernel=kernel_kernel, **self.object_gp_model_config.dict())
        object_gp.set_mean_function(ObjectConstant())
        optimizer.set_candidate_generator(self.grammar_generator)
        optimizer.set_oracle(self.oracle)
        optimizer.set_model(object_gp)
        optimizer.sample_train_set(self.n_initial_bo)
        optimizer.maximize(self.n_steps_bo)
        kernel_expression_best = optimizer.get_current_best()
        logger.info("Best kernel: " + str(kernel_expression_best))
        self.model = GPModel(kernel=kernel_expression_best.get_kernel(), **self.final_model_config.dict())
        self.model.infer(x_data, y_data)

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        return self.model.predictive_dist(x_test)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        return self.model.predictive_log_likelihood(x_test, y_test)

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        return self.model.entropy_predictive_dist(x_test)

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        return self.model.estimate_model_evidence(x_data, y_data)

    def reset_model(self):
        self.model = None
