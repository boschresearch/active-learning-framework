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

from copy import deepcopy
from alef.configs.kernels.hhk_configs import HHKEightLocalDefaultConfig
from alef.configs.models.ahgp_model_config import AHGPModelConfig
from alef.configs.models.base_model_config import BaseModelConfig
from alef.configs.models.deep_gp_config import DeepGPConfig
from alef.configs.models.gp_model_amortized_ensemble_config import BasicGPModelAmortizedEnsembleConfig
from alef.configs.models.gp_model_amortized_structured_config import BasicGPModelAmortizedStructuredConfig
from alef.configs.models.gp_model_config import BasicGPModelConfig
from alef.configs.models.gp_model_kernel_search_config import BaseGPModelKernelSearchConfig
from alef.configs.models.gp_model_marginalized_config import BasicGPModelMarginalizedConfig
from alef.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from alef.configs.models.gp_model_pytorch_ensemble_config import BasicGPModelPytorchEnsembleConfig
from alef.configs.models.mogp_model_config import BasicMOGPModelConfig
from alef.configs.models.svgp_model_pytorch_config import BasicSVGPModelPytorchConfig
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from alef.models.ahgp_model import AHGPModel
from alef.models.deep_gp import DeepGP
from alef.models.gp_model import GPModel
from alef.models.gp_model_amortized_structured import GPModelAmortizedStructured
from alef.models.gp_model_kernel_search import GPModelKernelSearch
from alef.models.gp_model_marginalized import GPModelMarginalized
from alef.models.mogp_model import MOGPModel
from alef.models.gp_model_laplace import GPModelLaplace
from alef.kernels.kernel_factory import KernelFactory
from alef.models.svgp_model_pytorch import SVGPModelPytorch
from alef.utils.gpflow_addon.multi_variance_likelihood import MultiGaussian
from alef.models.svgp_model import SVGPModel
from alef.configs.models.svgp_model_config import BasicSVGPConfig
from alef.models.gp_model_mcmc import GPModelMCMC
from alef.configs.models.gp_model_mcmc_config import (
    AdditiveProposalGPModelMCMCConfig,
    KernelGrammarGPModelMCMCConfig,
)
from alef.models.gp_model_mcmc_proposals import AdditiveGPMCMCProposal, AdditiveGPMCMCState
from alef.kernels.additive_kernel import Partition
from alef.models.gp_model_mcmc_proposals import (
    KernelGrammarMCMCProposal,
)
from alef.models.sparse_gp_model import SparseGpModel
from alef.configs.models.sparse_gp_model_config import BasicSparseGPModelConfig
from alef.models.gp_model_scalable import GPModelScalable
from alef.configs.models.gp_model_scalable_config import BasicScalableGPModelConfig
from alef.models.object_gp_model import ObjectGpModel
from alef.configs.models.object_gp_model_config import BasicObjectGPModelConfig
from alef.kernels.kernel_grammar.generator_factory import GeneratorFactory
from alef.configs.models.mogp_model_so_config import BasicSOMOGPModelConfig
from alef.models.mogp_model_so import SOMOGPModel
from alef.configs.models.mogp_model_so_marginalized_config import BasicSOMOGPModelMarginalizedConfig
from alef.models.mogp_model_so_marginalized import SOMOGPModelMarginalized
from alef.configs.models.gp_model_pytorch_config import BasicGPModelPytorchConfig
from alef.models.gp_model_pytorch import GPModelPytorch
from alef.models.gp_model_pytorch_ensemble import GPModelPytorchEnsemble
from alef.models.gp_model_amortized_ensemble import GPModelAmortizedEnsemble
from alef.configs.models.mogp_model_transfer_config import BasicTransferGPModelConfig
from alef.models.mogp_model_transfer import TransferGPModel
from alef.configs.models.metagp_model_config import BasicMetaGPModelConfig
from alef.models.metagp_model import MetaGPModel
import gpflow
import numpy as np


class ModelFactory:
    @staticmethod
    def build(model_config: BaseModelConfig):
        if isinstance(model_config, BasicGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicGPModelMarginalizedConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModelMarginalized(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicSparseGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = SparseGpModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicGPModelLaplaceConfig):
            # @TODO: all kernels should be applicaple to laplace and gpmodelmarg
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModelLaplace(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicSVGPConfig):
            kernel = gpflow.kernels.SharedIndependent(
                gpflow.kernels.SquaredExponential(), output_dim=model_config.output_dimension
            )
            lik = MultiGaussian(np.array([1.0 for _ in range(model_config.output_dimension)]))
            model = SVGPModel(kernel, lik, M=5, input_dim=model_config.input_dimension)
            model.set_optimizer(opt="adam", MAXITER=500, learning_rate=0.1)
            return model
        elif isinstance(model_config, BasicMOGPModelConfig):
            # kernel = gpflow.kernels.SharedIndependent(gpflow.kernels.SquaredExponential(), output_dim=model_config.output_dimension)
            kernel = KernelFactory.build(model_config.kernel_config)
            model = MOGPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicSOMOGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = SOMOGPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicTransferGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = TransferGPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicSOMOGPModelMarginalizedConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = SOMOGPModelMarginalized(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, AdditiveProposalGPModelMCMCConfig):
            partition = Partition(model_config.input_dimension)
            partition.set_partition_list(model_config.initial_partition_list)
            initial_proposal_state = AdditiveGPMCMCState(partition)
            proposal = AdditiveGPMCMCProposal(initial_proposal_state)
            model = GPModelMCMC(proposal=proposal, **model_config.dict())
            return model
        elif isinstance(model_config, KernelGrammarGPModelMCMCConfig):
            assert model_config.input_dimension == model_config.initial_base_kernel_config.input_dimension
            proposal = KernelGrammarMCMCProposal(model_config.initial_base_kernel_config, model_config.add_hp_prior)
            model = GPModelMCMC(proposal=proposal, **model_config.dict())
            return model
        elif isinstance(model_config, DeepGPConfig):
            model = DeepGP(**model_config.dict())
            return model
        elif isinstance(model_config, BasicScalableGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModelScalable(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicObjectGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = ObjectGpModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BaseGPModelKernelSearchConfig):
            grammar_generator_config = model_config.grammar_generator_config
            grammar_generator_config.input_dimension = model_config.input_dimension
            grammar_generator = GeneratorFactory.build(grammar_generator_config)
            model = GPModelKernelSearch(grammar_generator, **model_config.dict())
            return model
        elif isinstance(model_config, AHGPModelConfig):
            model = AHGPModel(**model_config.dict())
            return model
        elif isinstance(model_config, BasicGPModelPytorchConfig):
            kernel = PytorchKernelFactory.build(model_config.kernel_config)
            model = GPModelPytorch(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicMetaGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = MetaGPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicSVGPModelPytorchConfig):
            kernel = PytorchKernelFactory.build(model_config.kernel_config)
            model = SVGPModelPytorch(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicGPModelAmortizedStructuredConfig):
            kernel_config = model_config.kernel_config
            kernel_list = kernel_config.kernel_list
            model = GPModelAmortizedStructured(
                model_config.prediction_quantity,
                model_config.amortized_model_config,
                model_config.do_warm_start,
                model_config.warm_start_steps,
                model_config.warm_start_lr,
            )
            model.set_kernel_list(kernel_list)
            model.load_amortized_model(model_config.checkpoint_path, True)
            return model
        elif isinstance(model_config, BasicGPModelPytorchEnsembleConfig):
            model = GPModelPytorchEnsemble(gp_model_config=model_config.gp_model_config)
            return model
        elif isinstance(model_config, BasicGPModelAmortizedEnsembleConfig):
            kernel_list = model_config.kernel_list
            model = GPModelAmortizedEnsemble(
                model_config.prediction_quantity,
                model_config.amortized_model_config,
                model_config.entropy_approximation,
            )
            model.set_kernel_list(kernel_list)
            model.load_amortized_model(model_config.checkpoint_path, True)
            return model
        else:
            raise NotImplementedError(f"Invalid config: {model_config.__class__.__name__}")

    @staticmethod
    def change_input_dimension(model_config: BaseModelConfig, input_dimension=int) -> BaseModelConfig:
        transformed_model_config = deepcopy(model_config)
        if hasattr(transformed_model_config, "input_dimension"):
            transformed_model_config.input_dimension = input_dimension
        if hasattr(transformed_model_config, "kernel_config") and hasattr(
            transformed_model_config.kernel_config, "input_dimension"
        ):
            transformed_model_config.kernel_config.input_dimension = input_dimension
        return transformed_model_config


if __name__ == "__main__":
    kernel_config = HHKEightLocalDefaultConfig(input_dimension=2)
    config = BasicGPModelMarginalizedConfig(kernel_config=kernel_config, observation_noise=0.01)
    print(config.dict())
    new_config = ModelFactory.change_input_dimension(config, 3)
    print(new_config.dict())
    print(config.dict())
