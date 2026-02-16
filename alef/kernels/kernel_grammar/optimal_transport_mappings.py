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

from alef.configs.kernels.linear_configs import LinearWithPriorConfig
from alef.configs.kernels.matern32_configs import Matern32WithPriorConfig
from alef.configs.kernels.matern52_configs import Matern52WithPriorConfig
from alef.configs.kernels.periodic_configs import (
    PeriodicWithPriorConfig,
    PeriodicWithPriorSmallerInitialPeriodicityConfig,
)
from alef.configs.kernels.rational_quadratic_configs import RQWithPriorConfig
from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
from alef.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
from alef.kernels.linear_kernel import LinearKernel
from alef.kernels.rbf_kernel import RBFKernel


class KernelGrammarTreeDistanceMapper:
    def __init__(self, generator_name: str, input_dimension: int) -> None:
        self.create_mapping_dict(generator_name, input_dimension)

    def create_mapping_dict(self, generator_name: str, input_dimension: int):
        assert generator_name is not None
        self.mapping_dict = {}
        if generator_name == "CompositionalKernelSearchSpace":
            base_name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            base_name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            base_name_per_kernel = PeriodicWithPriorConfig(input_dimension=0).name
            for i in range(0, input_dimension):
                suffix = "_on_" + str(i)
                name_rbf_kernel = base_name_rbf_kernel + suffix
                name_linear_kernel = base_name_linear_kernel + suffix
                name_per_kernel = base_name_per_kernel + suffix
                self.mapping_dict[str([name_rbf_kernel, name_per_kernel])] = (1.0, [name_rbf_kernel, name_per_kernel])
                self.mapping_dict[str([name_rbf_kernel])] = (1.0, [name_rbf_kernel])
                self.mapping_dict[str([name_linear_kernel])] = (1.0, [name_linear_kernel])
                self.mapping_dict[str([name_per_kernel])] = (1.0, [name_per_kernel])

        elif generator_name == "NDimFullKernelsSearchSpace":
            name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            name_per_kernel = PeriodicWithPriorConfig(input_dimension=0).name
            self.mapping_dict = {}
            self.mapping_dict[str([name_rbf_kernel, name_per_kernel])] = (1.0, [name_rbf_kernel, name_per_kernel])
            self.mapping_dict[str([name_rbf_kernel])] = (1.0, [name_rbf_kernel])
            self.mapping_dict[str([name_linear_kernel])] = (1.0, [name_linear_kernel])
            self.mapping_dict[str([name_per_kernel])] = (1.0, [name_per_kernel])
        elif generator_name == "CKSWithRQSearchSpace":
            base_name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            base_name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            base_name_per_kernel = PeriodicWithPriorConfig(input_dimension=0).name
            base_name_rq_kernel = RQWithPriorConfig(input_dimension=0).name
            for i in range(0, input_dimension):
                suffix = "_on_" + str(i)
                name_rbf_kernel = base_name_rbf_kernel + suffix
                name_linear_kernel = base_name_linear_kernel + suffix
                name_per_kernel = base_name_per_kernel + suffix
                name_rq_kernel = base_name_rq_kernel + suffix
                self.mapping_dict[str([name_rbf_kernel, name_per_kernel, name_rq_kernel])] = (
                    0.2,
                    [name_rbf_kernel, name_per_kernel, name_rq_kernel],
                )
                self.mapping_dict[str([name_rbf_kernel, name_rq_kernel])] = (1.0, [name_rbf_kernel, name_rq_kernel])
                self.mapping_dict[str([name_rbf_kernel])] = (0.2, [name_rbf_kernel])
                self.mapping_dict[str([name_linear_kernel])] = (2.0, [name_linear_kernel])
                self.mapping_dict[str([name_per_kernel])] = (2.0, [name_per_kernel])
                self.mapping_dict[str([name_rq_kernel])] = (0.2, [name_rq_kernel])
        else:
            raise NotImplementedError

    def get_weighted_feature_dict(self, kernel_expression: BaseKernelGrammarExpression):
        count_dict = self.get_count_dict(kernel_expression, True)
        assert len(count_dict) > 0
        weight_dict = self.get_weight_dict()
        feature_dict = {}
        for key in count_dict:
            feature_dict[key] = weight_dict[key] * count_dict[key]
        return feature_dict

    def get_weight_dict(self):
        weight_dict = {}
        for key in self.mapping_dict:
            weight_dict[key] = self.mapping_dict[key][0]
        return weight_dict

    def get_count_dict(self, kernel_expression: BaseKernelGrammarExpression, normalize=True):
        elementary_count_dict = kernel_expression.get_elementary_count_dict()
        number_elementaries = kernel_expression.count_elementary_expressions()
        count_dict = {}
        for key in self.mapping_dict:
            elementary_name_list = self.mapping_dict[key][1]
            for elementary_name in elementary_name_list:
                if elementary_name in elementary_count_dict:
                    if key in count_dict:
                        count_dict[key] += elementary_count_dict[elementary_name]
                    else:
                        count_dict[key] = elementary_count_dict[elementary_name]
        if normalize:
            for key in count_dict:
                count_dict[key] = float(count_dict[key]) / float(number_elementaries)
        return count_dict


class DimWiseWeightedDistanceExtractor:
    def __init__(self, generator_name: str, input_dimension: int) -> None:
        self.create_dim_class_dict(generator_name, input_dimension)
        self.empty_feature_präfix = "NULL"
        if (
            generator_name == "CompositionalKernelSearchSpace"
            or generator_name == "CKSWithRQSearchSpace"
            or generator_name == "CKSHighDimSearchSpace"
            or generator_name == "CKSWithRQTimeSeriesSearchSpace"
            or generator_name == "CKSTimeSeriesSearchSpace"
        ):
            self.n_differentiated_dimensions = input_dimension
        else:
            self.n_differentiated_dimensions = 1

    def create_dim_class_dict(self, generator_name: str, input_dimension: int):
        assert generator_name is not None
        self.dim_class_dict = {}
        if generator_name == "CompositionalKernelSearchSpace":
            base_name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            base_name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            base_name_per_kernel = PeriodicWithPriorConfig(input_dimension=0).name
            for i in range(0, input_dimension):
                suffix = "_on_" + str(i)
                name_rbf_kernel = base_name_rbf_kernel + suffix
                name_linear_kernel = base_name_linear_kernel + suffix
                name_per_kernel = base_name_per_kernel + suffix
                self.dim_class_dict[i] = [name_rbf_kernel, name_per_kernel, name_linear_kernel]
        elif generator_name == "CKSWithRQSearchSpace":
            base_name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            base_name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            base_name_per_kernel = PeriodicWithPriorConfig(input_dimension=0).name
            base_name_rq_kernel = RQWithPriorConfig(input_dimension=0).name
            for i in range(0, input_dimension):
                suffix = "_on_" + str(i)
                name_rbf_kernel = base_name_rbf_kernel + suffix
                name_linear_kernel = base_name_linear_kernel + suffix
                name_per_kernel = base_name_per_kernel + suffix
                name_rq_kernel = base_name_rq_kernel + suffix
                self.dim_class_dict[i] = [name_rbf_kernel, name_per_kernel, name_linear_kernel, name_rq_kernel]
        elif generator_name == "CKSWithRQTimeSeriesSearchSpace":
            base_name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            base_name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            base_name_per_kernel = PeriodicWithPriorSmallerInitialPeriodicityConfig(input_dimension=0).name
            base_name_rq_kernel = RQWithPriorConfig(input_dimension=0).name
            for i in range(0, input_dimension):
                suffix = "_on_" + str(i)
                name_rbf_kernel = base_name_rbf_kernel + suffix
                name_linear_kernel = base_name_linear_kernel + suffix
                name_per_kernel = base_name_per_kernel + suffix
                name_rq_kernel = base_name_rq_kernel + suffix
                self.dim_class_dict[i] = [name_rbf_kernel, name_per_kernel, name_linear_kernel, name_rq_kernel]
        elif generator_name == "CKSTimeSeriesSearchSpace":
            base_name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            base_name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            base_name_per_kernel = PeriodicWithPriorSmallerInitialPeriodicityConfig(input_dimension=0).name
            for i in range(0, input_dimension):
                suffix = "_on_" + str(i)
                name_rbf_kernel = base_name_rbf_kernel + suffix
                name_linear_kernel = base_name_linear_kernel + suffix
                name_per_kernel = base_name_per_kernel + suffix
                self.dim_class_dict[i] = [name_rbf_kernel, name_per_kernel, name_linear_kernel]
        elif generator_name == "CKSHighDimSearchSpace":
            base_name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            base_name_rq_kernel = RQWithPriorConfig(input_dimension=0).name
            for i in range(0, input_dimension):
                suffix = "_on_" + str(i)
                name_rbf_kernel = base_name_rbf_kernel + suffix
                name_rq_kernel = base_name_rq_kernel + suffix
                self.dim_class_dict[i] = [name_rbf_kernel, name_rq_kernel]
        elif generator_name == "NDimFullKernelsSearchSpace":
            name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            name_per_kernel = PeriodicWithPriorConfig(input_dimension=0).name
            self.dim_class_dict = {}
            self.dim_class_dict[0] = [name_rbf_kernel, name_per_kernel, name_linear_kernel]
        elif (
            generator_name == "BigNDimFullKernelsSearchSpace" or generator_name == "BigLocalNDimFullKernelsSearchSpace"
        ):
            name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            name_rq_kernel = RQWithPriorConfig(input_dimension=0).name
            name_matern52_kernel = Matern52WithPriorConfig(input_dimension=0).name
            name_matern32_kernel = Matern32WithPriorConfig(input_dimension=0).name
            self.dim_class_dict = {}
            self.dim_class_dict[0] = [
                name_rbf_kernel,
                name_rq_kernel,
                name_linear_kernel,
                name_matern32_kernel,
                name_matern52_kernel,
            ]
        elif generator_name == "DynamicHierarchicalHyperplaneKernelSpace":
            name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            self.dim_class_dict = {}
            self.dim_class_dict[0] = [name_rbf_kernel]
        elif generator_name == "FlatLocalKernelSearchSpace":
            name_rbf_kernel = RBFWithPriorConfig(input_dimension=0).name
            name_linear_kernel = LinearWithPriorConfig(input_dimension=0).name
            name_matern_kernel = Matern52WithPriorConfig(input_dimension=0).name
            self.dim_class_dict = {}
            self.dim_class_dict[0] = [name_rbf_kernel, name_matern_kernel, name_linear_kernel]
        else:
            raise NotImplementedError("Kernel search space not considered in optimal_transport_mappings")

    def get_dim_wise_weighted_elementary_features(self, kernel_expression: BaseKernelGrammarExpression):
        elementary_count_dict = kernel_expression.get_elementary_count_dict()
        feature_dict = {}
        for i in range(0, self.n_differentiated_dimensions):
            counter = 0
            null_feature = self.empty_feature_präfix + "_" + str(i)
            for elementary_name in self.dim_class_dict[i]:
                if elementary_name in elementary_count_dict:
                    counter += elementary_count_dict[elementary_name]
            if counter == 0:
                feature_dict[null_feature] = 1.0
                for elementary_name in self.dim_class_dict[i]:
                    feature_dict[elementary_name] = 0.0
            else:
                feature_dict[null_feature] = 0.0
                for elementary_name in self.dim_class_dict[i]:
                    if elementary_name in elementary_count_dict:
                        feature_dict[elementary_name] = float(elementary_count_dict[elementary_name]) / float(counter)
                    else:
                        feature_dict[elementary_name] = 0.0
        return feature_dict


if __name__ == "__main__":
    base_expression_1 = ElementaryKernelGrammarExpression(
        RBFKernel(**RBFWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict())
    )
    base_expression_2 = ElementaryKernelGrammarExpression(
        LinearKernel(
            **LinearWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict()
        )
    )
    base_expression_3 = ElementaryKernelGrammarExpression(
        LinearKernel(
            **LinearWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=0).dict()
        )
    )
    base_expression_4 = ElementaryKernelGrammarExpression(
        RBFKernel(**RBFWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1).dict())
    )

    if True:
        expression = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
        expression2 = KernelGrammarExpression(expression, base_expression_3, KernelGrammarOperator.ADD)
        expression3 = KernelGrammarExpression(base_expression_4, expression2, KernelGrammarOperator.MULTIPLY)
        expression4 = KernelGrammarExpression(expression2, expression3, KernelGrammarOperator.MULTIPLY)

    print(expression3)
    print(expression4)
    print("")
    print(expression3.get_elementary_count_dict())
    print("")
    print(expression4.get_elementary_count_dict())

    mapper = DimWiseWeightedDistanceExtractor("CompositionalKernelSearchSpace", 3)
    print(mapper.get_dim_wise_weighted_elementary_features(expression4))
