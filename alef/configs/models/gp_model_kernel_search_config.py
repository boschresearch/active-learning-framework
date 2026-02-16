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

from typing import Tuple
from pydantic import BaseSettings

from alef.configs.kernels.kernel_grammar_generators.base_kernel_grammar_generator_config import (
    BaseKernelGrammarGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.cks_high_dim_generator_config import CKSHighDimGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import (
    CKSWithRQGeneratorConfig,
    CKSWithRQTimeSeriesGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.compositional_kernel_search_configs import (
    CKSTimeSeriesGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.local_kernel_search_generator_config import (
    BigLocalNDimFullKernelsGrammarGeneratorConfig,
    FlatLocalKernelSearchSpaceConfig,
)
from alef.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import (
    BigNDimFullKernelsGrammarGeneratorConfig,
)
from alef.models.gp_model_kernel_search import OracleType


class BaseGPModelKernelSearchConfig(BaseSettings):
    input_dimension: int
    grammar_generator_config: BaseKernelGrammarGeneratorConfig
    oracle_type: OracleType = OracleType.BIC
    fast_inference: bool = False
    n_steps_bo: int = 50
    use_meta_gp_hp_prior: bool = False
    default_n_steps_ea: int = 6
    population_size_tuple: Tuple[int, int] = (50, 100)
    bo_n_initial_factor: int = 3


class GPKernelSearchCKSwithRQ(BaseGPModelKernelSearchConfig):
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = CKSWithRQGeneratorConfig(input_dimension=0)


class GPKernelSearchCKSwithRQTimeSeries(BaseGPModelKernelSearchConfig):
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = CKSWithRQTimeSeriesGeneratorConfig(input_dimension=0)
    default_n_steps_ea: int = 4
    use_meta_gp_hp_prior: bool = True


class GPKernelSearchCKSTimeSeries(BaseGPModelKernelSearchConfig):
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = CKSTimeSeriesGeneratorConfig(input_dimension=0)
    default_n_steps_ea: int = 4
    use_meta_gp_hp_prior: bool = True


class GPKernelSearchCKSwithHighDim(BaseGPModelKernelSearchConfig):
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = CKSHighDimGeneratorConfig(input_dimension=0)


class GPFlatLocalKernelSearchConfig(BaseGPModelKernelSearchConfig):
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = FlatLocalKernelSearchSpaceConfig(input_dimension=0)
    n_steps_bo: int = 30
    default_n_steps_ea: int = 4


class GPBigLocalNDimFullKernelSearchConfig(BaseGPModelKernelSearchConfig):
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = BigLocalNDimFullKernelsGrammarGeneratorConfig(
        input_dimension=0
    )


class GPBigNDimFullKernelSearchConfig(BaseGPModelKernelSearchConfig):
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = BigNDimFullKernelsGrammarGeneratorConfig(
        input_dimension=0
    )


class GPKernelSearchCKSwithRQEvidence(BaseGPModelKernelSearchConfig):
    oracle_type: OracleType = OracleType.EVIDENCE
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = CKSWithRQGeneratorConfig(input_dimension=0)


class GPKernelSearchCKSwithHighDimEvidence(BaseGPModelKernelSearchConfig):
    oracle_type: OracleType = OracleType.EVIDENCE
    grammar_generator_config: BaseKernelGrammarGeneratorConfig = CKSHighDimGeneratorConfig(input_dimension=0)
