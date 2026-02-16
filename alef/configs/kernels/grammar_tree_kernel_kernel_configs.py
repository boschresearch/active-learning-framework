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

from typing import List, Tuple
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.kernels.kernel_kernel_grammar_tree import FeatureType


class KernelGrammarSubtreeKernelConfig(BaseKernelConfig):
    input_dimension: int = 0
    base_variance: float = 1.0
    parameters_trainable: bool = True
    transform_to_normal: bool = False
    name = "SubTreeKernelKernel"


class OptimalTransportGrammarKernelConfig(BaseKernelConfig):
    input_dimension: int = 0
    feature_type_list: List[FeatureType] = [FeatureType.ELEMENTARY_COUNT, FeatureType.SUBTREES]
    base_variance: float = 1.0
    base_lengthscale: float = 1.0
    base_alpha: float = 0.5
    alpha_trainable: bool = True
    parameters_trainable: bool = True
    transform_to_normal: bool = False
    use_hyperprior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = (0.1, 0.7)
    variance_prior_parameters: Tuple[float, float] = (0.4, 0.7)
    name = "OptimalTransportGrammarKernel"


class TreeBasedOTGrammarKernelConfig(OptimalTransportGrammarKernelConfig):
    feature_type_list: List[FeatureType] = [FeatureType.ONE_GRAM_TREE_METRIC, FeatureType.SUBTREES]
    name = "TreeBasedOTGrammarKernelConfig"


class OTWeightedDimsExtendedGrammarKernelConfig(OptimalTransportGrammarKernelConfig):
    feature_type_list: List[FeatureType] = [
        FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT,
        FeatureType.REDUCED_ELEMENTARY_PATHS,
        FeatureType.SUBTREES,
    ]
    name = "OTWeightedDimsExtendedGrammarKernel"


class OTWeightedDimsExtendedKernelWithHyperpriorConfig(OTWeightedDimsExtendedGrammarKernelConfig):
    use_hyperprior: bool = True
    name = "OTWeightedDimsExtendedKernelWithHyperprior"


class OTWeightedDimsInvarianceGrammarKernelConfig(OptimalTransportGrammarKernelConfig):
    feature_type_list: List[FeatureType] = [
        FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT,
        FeatureType.REDUCED_ELEMENTARY_PATHS,
        FeatureType.ADD_MULT_INVARIANT_SUBTREES,
    ]
    transform_to_normal: bool = True
    name = "OTWeightedDimsInvarianceGrammarKernel"
