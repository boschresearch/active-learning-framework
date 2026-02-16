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

from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.grammar_tree_kernel_kernel_configs import (
    KernelGrammarSubtreeKernelConfig,
    OptimalTransportGrammarKernelConfig,
)
from alef.configs.kernels.kernel_list_configs import BasicKernelListConfig, SEKernelViaKernelListConfig
from alef.configs.kernels.matern32_configs import BasicMatern32Config
from alef.configs.kernels.rational_quadratic_configs import BasicRQConfig
from alef.configs.kernels.spectral_mixture_kernel_config import BasicSMKernelConfig
from alef.configs.kernels.wasi_config import BasicWasiConfig
from alef.configs.kernels.weighted_additive_kernel_config import BasicWeightedAdditiveKernelConfig
from alef.kernels.deep_kernels.base_deep_kernel import BaseDeepKernel
from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from alef.kernels.kernel_kernel_grammar_tree import (
    KernelGrammarSubtreeKernel,
    OptimalTransportKernelKernel,
)
from alef.configs.kernels.wami_configs import BasicWamiConfig
from alef.configs.kernels.hhk_configs import BasicHHKConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig
from alef.kernels.hierarchical_hyperplane_kernel import HKKInputInitializedBaseKernel, HierarchicalHyperplaneKernel
from alef.kernels.rational_quadratic_kernel import RationalQuadraticKernel
from alef.kernels.spectral_mixture_kernel import SpectralMixtureKernel
from alef.kernels.warped_multi_index_kernel import WarpedMultiIndexKernel
from alef.kernels.rbf_kernel import RBFKernel
from alef.kernels.neural_kernel_network import NeuralKernelNetwork
from alef.configs.kernels.neural_kernel_network_config import BasicNKNConfig
from alef.kernels.additive_kernel import AdditiveKernel
from alef.configs.kernels.additive_kernel_configs import BasicAdditiveKernelConfig
from alef.configs.kernels.matern52_configs import BasicMatern52Config
from alef.kernels.matern52_kernel import Matern52Kernel
from alef.kernels.matern32_kernel import Matern32Kernel
from alef.kernels.linear_kernel import LinearKernel
from alef.configs.kernels.linear_configs import BasicLinearConfig
from alef.kernels.rbf_object_kernel import RBFObjectKernel
from alef.configs.kernels.rbf_object_configs import BasicRBFObjectConfig
from alef.configs.kernels.hellinger_kernel_kernel_configs import BasicHellingerKernelKernelConfig
from alef.kernels.kernel_kernel_hellinger import KernelKernelHellinger
from alef.kernels.periodic_kernel import PeriodicKernel
from alef.configs.kernels.periodic_configs import BasicPeriodicConfig
from alef.configs.kernels.deep_kernels.base_deep_kernel_config import BaseDeepKernelConfig
from alef.kernels.warped_single_index_kernel import WarpedSingleIndexKernel
from alef.kernels.weighted_additive_kernel import WeightedAdditiveKernel
from alef.models.feature_extractors.feature_extractor_factory import FeatureExtractorFactory
from alef.configs.kernels.multi_output_kernels.coregionalization_kernel_configs import (
    BasicCoregionalizationSOConfig,
    BasicCoregionalizationMOConfig,
)
from alef.kernels.multi_output_kernels.coregionalization_kernel import (
    CoregionalizationSOKernel,
    CoregionalizationMOKernel,
)
from alef.configs.kernels.multi_output_kernels.coregionalization_1latent_kernel_configs import (
    BasicCoregionalization1LConfig,
)
from alef.kernels.multi_output_kernels.coregionalization_1latent_kernel import Coregionalization1LKernel
from alef.configs.kernels.multi_output_kernels.coregionalization_Platent_kernel_configs import (
    BasicCoregionalizationPLConfig,
)
from alef.kernels.multi_output_kernels.coregionalization_Platent_kernel import CoregionalizationPLKernel
from alef.configs.kernels.multi_output_kernels.multi_source_additive_kernel_configs import BasicMIAdditiveConfig
from alef.kernels.multi_output_kernels.multi_source_additive_kernel import MIAdditiveKernel
from alef.configs.kernels.multi_output_kernels.coregionalization_transfer_kernel_config import (
    BasicCoregionalizationTransferConfig,
)
from alef.kernels.multi_output_kernels.coregionalization_transfer_kernel import CoregionalizationTransferKernel
from alef.configs.kernels.multi_output_kernels.flexible_transfer_kernel_config import BasicFlexibleTransferConfig
from alef.kernels.multi_output_kernels.flexible_transfer_kernel import FlexibleTransferKernel
from alef.configs.kernels.multi_output_kernels.fpacoh_kernel_config import BasicFPACOHKernelConfig
from alef.kernels.multi_output_kernels.fpacoh_kernel import FPACOHKernel


class KernelFactory:
    @staticmethod
    def build(kernel_config: BaseKernelConfig):
        if isinstance(kernel_config, BasicHHKConfig):
            base_kernel_config = kernel_config.base_kernel_config
            base_kernel_config.input_dimension = kernel_config.input_dimension
            base_kernel = KernelFactory.build(base_kernel_config)
            if isinstance(base_kernel, InputInitializedKernelInterface):
                # Used mainly to use HHK in combination with SM kernel
                kernel = HKKInputInitializedBaseKernel(base_kernel=base_kernel, **kernel_config.dict())
            else:
                kernel = HierarchicalHyperplaneKernel(base_kernel=base_kernel, **kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicWamiConfig):
            kernel = WarpedMultiIndexKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicRBFConfig):
            kernel = RBFKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicNKNConfig):
            kernel = NeuralKernelNetwork(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicAdditiveKernelConfig):
            kernel = AdditiveKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicMatern52Config):
            kernel = Matern52Kernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicMatern32Config):
            kernel = Matern32Kernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicLinearConfig):
            kernel = LinearKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BaseDeepKernelConfig):
            feature_extractor = FeatureExtractorFactory.build(
                kernel_config.feature_extractor_config, kernel_config.input_dimension
            )
            kernel = BaseDeepKernel(feature_extractor=feature_extractor, **kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicRBFObjectConfig):
            kernel = RBFObjectKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicRQConfig):
            kernel = RationalQuadraticKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicPeriodicConfig):
            kernel = PeriodicKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicHellingerKernelKernelConfig):
            kernel = KernelKernelHellinger(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, KernelGrammarSubtreeKernelConfig):
            kernel = KernelGrammarSubtreeKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, OptimalTransportGrammarKernelConfig):
            kernel = OptimalTransportKernelKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicCoregionalizationSOConfig):
            kernel = CoregionalizationSOKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicCoregionalizationMOConfig):
            kernel = CoregionalizationMOKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicCoregionalization1LConfig):
            kernel = Coregionalization1LKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicCoregionalizationPLConfig):
            kernel = CoregionalizationPLKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicWeightedAdditiveKernelConfig):
            base_kernel_config_list = kernel_config.base_kernel_config_list
            input_dimension = kernel_config.input_dimension
            base_kernel_list = []
            for base_kernel_config in base_kernel_config_list:
                base_kernel_config.input_dimension = input_dimension
                base_kernel = KernelFactory.build(base_kernel_config)
                base_kernel_list.append(base_kernel)
            kernel = WeightedAdditiveKernel(base_kernel_list=base_kernel_list, **kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicSMKernelConfig):
            kernel = SpectralMixtureKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicKernelListConfig):
            from alef.models.amortized_infer_structured_kernels.gp.base_kernels import (
                transform_kernel_list_to_expression,
            )

            expanded_kernel_list = [kernel_config.kernel_list for i in range(0, kernel_config.input_dimension)]
            kernel_expression = transform_kernel_list_to_expression(
                expanded_kernel_list, add_prior=kernel_config.add_prior, use_gpflow=True
            )
            return kernel_expression.get_kernel()
        elif isinstance(kernel_config, BasicMIAdditiveConfig):
            kernel = MIAdditiveKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicCoregionalizationTransferConfig):
            kernel = CoregionalizationTransferKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicFlexibleTransferConfig):
            kernel = FlexibleTransferKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicFPACOHKernelConfig):
            kernel = FPACOHKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicWasiConfig):
            kernel = WarpedSingleIndexKernel(**kernel_config.dict())
            return kernel
        else:
            raise NotImplementedError("Invalid config")


if __name__ == "__main__":
    config = SEKernelViaKernelListConfig(input_dimension=2)
    print(config.dict())
    kernel = KernelFactory.build(config)
    print(kernel)
    # print(kernel)
