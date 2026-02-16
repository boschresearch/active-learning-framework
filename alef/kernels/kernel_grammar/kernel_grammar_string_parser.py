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

from alef.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
from alef.kernels.kernel_grammar.kernel_grammar_search_spaces import (
    BaseKernelGrammarSearchSpace,
    CKSTimeSeriesSearchSpace,
)


class ExpressionStringParser:
    def __init__(self, search_space: BaseKernelGrammarSearchSpace):
        self.search_space = search_space
        self.init_base_kernel_dict()
        self.operator_dict = {}
        self.operator_dict["MULTIPLY"] = KernelGrammarOperator.MULTIPLY
        self.operator_dict["ADD"] = KernelGrammarOperator.ADD
        self.operator_dict["SPLIT_CH"] = KernelGrammarOperator.SPLIT_CH
        self.replace_tag_prefix = "REPLACE"

    def init_base_kernel_dict(self):
        base_expression_list = self.search_space.get_root_expressions()
        self.base_kernel_dict = {}
        for root_expression in base_expression_list:
            self.base_kernel_dict[root_expression.get_name()] = root_expression

    def parse(self, expression_string: str) -> BaseKernelGrammarExpression:
        subexpression_string_dict, last_index = self.expression_string_parser(expression_string)
        return self.kernel_expression_from_dict_resolver(subexpression_string_dict, last_index)

    def expression_string_parser(self, expression_string: str):
        i = 0
        replaced_expression_string = expression_string
        subexpression_string_dict = {}
        has_subexpression = True
        while has_subexpression:
            replace_tag = "{}_{}".format(self.replace_tag_prefix, i)
            sub_expression_string, replaced_expression_string, has_subexpression = (
                self.extract_and_replace_sub_expression(replaced_expression_string, replace_tag)
            )
            if has_subexpression:
                subexpression_string_dict[replace_tag] = self.parse_subexpression_string(sub_expression_string)
                i += 1
        last_index = i - 1
        return subexpression_string_dict, last_index

    def kernel_expression_from_dict_resolver(
        self, subexpression_string_dict, index_replace_tag: int
    ) -> BaseKernelGrammarExpression:
        replace_tag = "{}_{}".format(self.replace_tag_prefix, index_replace_tag)
        subexpression_tuple = subexpression_string_dict[replace_tag]
        sub_expression_string_0 = subexpression_tuple[0]
        sub_expression_string_1 = subexpression_tuple[2]
        sub_expression_operator_string = subexpression_tuple[1]
        if sub_expression_string_0 in self.base_kernel_dict:
            expression_0 = self.base_kernel_dict[sub_expression_string_0].deep_copy()
        else:
            index = sub_expression_string_0.split("_")[1]
            expression_0 = self.kernel_expression_from_dict_resolver(subexpression_string_dict, index)

        if sub_expression_string_1 in self.base_kernel_dict:
            expression_1 = self.base_kernel_dict[sub_expression_string_1].deep_copy()
        else:
            index = sub_expression_string_1.split("_")[1]
            expression_1 = self.kernel_expression_from_dict_resolver(subexpression_string_dict, index)

        expression = KernelGrammarExpression(
            expression_0, expression_1, self.operator_dict[sub_expression_operator_string]
        )
        return expression

    def parse_subexpression_string(self, subexpression_string: str):
        symbol_tuple = subexpression_string[1:-1].split(" ")
        assert len(symbol_tuple) == 3
        return symbol_tuple

    def extract_and_replace_sub_expression(self, expression_string: str, replace_tag: str):
        expression_start_index = 0
        expression_end_index = 0
        has_subexpression = False
        for index, letter in enumerate(expression_string):
            if letter == "(":
                expression_start_index = index
                has_subexpression = True
            elif letter == ")":
                expression_end_index = index + 1
                break
        sub_expression_string = expression_string[expression_start_index:expression_end_index]
        replaced_expression_string = (
            expression_string[:expression_start_index] + replace_tag + expression_string[expression_end_index:]
        )
        return sub_expression_string, replaced_expression_string, has_subexpression


if __name__ == "__main__":
    search_sapce = CKSTimeSeriesSearchSpace(4)
    expression_string = "(((((LinearWithPrior_on_0 MULTIPLY (((PeriodicWithPriorSmallerInitialPeriodicity_on_0 ADD PeriodicWithPriorSmallerInitialPeriodicity_on_0) ADD PeriodicWithPriorSmallerInitialPeriodicity_on_0) ADD PeriodicWithPriorSmallerInitialPeriodicity_on_0)) ADD RBFWithPrior_on_0) ADD LinearWithPrior_on_0) MULTIPLY PeriodicWithPriorSmallerInitialPeriodicity_on_0) ADD LinearWithPrior_on_0)"
    parser = ExpressionStringParser(search_sapce)
    expression = parser.parse(expression_string)
    print(expression)
    # expression_string_parser(expression_string)
