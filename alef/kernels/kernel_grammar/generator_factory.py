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
    CompositionalKernelSearchGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.dynamic_hhk_generator_config import DynamicHHKGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.local_kernel_search_generator_config import (
    BigLocalNDimFullKernelsGrammarGeneratorConfig,
    FlatLocalKernelSearchSpaceConfig,
)
from alef.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import (
    BigNDimFullKernelsGrammarGeneratorConfig,
    NDimFullKernelsGrammarGeneratorConfig,
)
from alef.kernels.kernel_grammar.kernel_grammar_search_spaces import (
    BigLocalNDimFullKernelsSearchSpace,
    BigNDimFullKernelsSearchSpace,
    CKSHighDimSearchSpace,
    CKSTimeSeriesSearchSpace,
    CKSWithRQTimeSeriesSearchSpace,
    DynamicHierarchicalHyperplaneKernelSpace,
    FlatLocalKernelSearchSpace,
    NDimFullKernelsSearchSpace,
    CKSWithRQSearchSpace,
    CompositionalKernelSearchSpace,
)
from alef.kernels.kernel_grammar.kernel_grammar_candidate_generator import KernelGrammarCandidateGenerator


class GeneratorFactory:
    @staticmethod
    def build(generator_config: BaseKernelGrammarGeneratorConfig):
        if isinstance(generator_config, NDimFullKernelsGrammarGeneratorConfig):
            search_space = NDimFullKernelsSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CompositionalKernelSearchGeneratorConfig):
            search_space = CompositionalKernelSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CKSWithRQGeneratorConfig):
            search_space = CKSWithRQSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CKSWithRQTimeSeriesGeneratorConfig):
            search_space = CKSWithRQTimeSeriesSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CKSTimeSeriesGeneratorConfig):
            search_space = CKSTimeSeriesSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CKSHighDimGeneratorConfig):
            search_space = CKSHighDimSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, DynamicHHKGeneratorConfig):
            search_space = DynamicHierarchicalHyperplaneKernelSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, FlatLocalKernelSearchSpaceConfig):
            search_space = FlatLocalKernelSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, BigNDimFullKernelsGrammarGeneratorConfig):
            search_space = BigNDimFullKernelsSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, BigLocalNDimFullKernelsGrammarGeneratorConfig):
            search_space = BigLocalNDimFullKernelsSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
