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

import logging
from typing import List

import numpy as np
from alef.bayesian_optimization.base_candidate_generator import CandidateGenerator
from alef.kernels.kernel_grammar.kernel_grammar_search_spaces import (
    BaseKernelGrammarSearchSpace,
    CompositionalKernelSearchSpace,
)

logger = logging.getLogger(__name__)


class KernelGrammarCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        search_space: BaseKernelGrammarSearchSpace,
        n_initial_factor_trailing: int,
        n_initial_flat_trailing: int,
        depth_initial_flat_trailing: int,
        do_flat_initial_trailing: bool,
        n_exploration_trailing: int,
        exploration_p_geometric: float,
        n_exploitation_trailing: int,
        walk_length_exploitation_trailing: int,
        do_random_walk_exploitation_trailing: bool,
        limit_n_additional_current_best: bool,
        **kwargs,
    ):
        self.search_space = search_space
        self.n_initial_trailing = n_initial_factor_trailing * self.search_space.get_num_base_kernels()
        self.n_initial_flat_trailing = n_initial_flat_trailing
        self.do_flat_initial_trailing = do_flat_initial_trailing
        self.depth_initial_flat_trailing = depth_initial_flat_trailing
        assert n_exploitation_trailing % walk_length_exploitation_trailing == 0
        self.n_exploitation_trailing = n_exploitation_trailing
        self.walk_length_exploitation_trailing = walk_length_exploitation_trailing
        self.n_exploration_trailing = n_exploration_trailing
        self.exploration_p_geometric = exploration_p_geometric
        self.do_random_walk_exploitation_trailing = do_random_walk_exploitation_trailing
        self.limit_n_additional_current_best = limit_n_additional_current_best

    def get_random_canditates(self, n_candidates: int, seed=100, set_seed=False) -> List[object]:
        """
        Generates random candidates by performing random walks from each root expressions in the search space
        """
        # if n==input_dimension and thus smaller than number of root expressions/base kernels only return input filling kernels
        if set_seed:
            np.random.seed(seed)
        if n_candidates <= self.search_space.input_dimension:
            dim_filling_expressions = self.search_space.get_dimension_filling_expressions()
            expressions = list(np.random.choice(dim_filling_expressions, n_candidates, replace=False))
            return expressions
        assert n_candidates % self.search_space.get_num_base_kernels() == 0
        depth = int((n_candidates / self.search_space.get_num_base_kernels()) - 1)
        random_candidates = []
        for root_expression in self.search_space.get_root_expressions():
            random_candidates.append(root_expression)
            random_candidates += self.search_space.random_walk(depth, root_expression)
        return random_candidates

    def get_random_candidates_flat(self, n_candidates: int, depth: int, seed=100, set_seed=False):
        """
        Generates random candidates up to a specified depth
        """
        # if n==input_dimension and thus smaller than number of root expressions/base kernels only return input filling kernels
        if set_seed:
            np.random.seed(seed)
        if n_candidates <= self.search_space.input_dimension:
            dim_filling_expressions = self.search_space.get_dimension_filling_expressions()
            expressions = list(np.random.choice(dim_filling_expressions, n_candidates, replace=False))
            return expressions
        root_expressions = self.search_space.get_root_expressions()
        random_candidates = [] + root_expressions
        while len(random_candidates) < n_candidates:
            root_expression = np.random.choice(root_expressions)
            random_walk = self.search_space.random_walk(depth, root_expression)
            for random_canidate in random_walk:
                if len(random_candidates) < n_candidates:
                    random_candidates.append(random_canidate)
        return random_candidates

    def get_initial_candidates_trailing(self) -> List[object]:
        """
        Returns the initial candidates for the trailing optimization in the Object-BO procedure
        """
        if self.do_flat_initial_trailing:
            return self.get_random_candidates_flat(self.n_initial_flat_trailing, self.depth_initial_flat_trailing)
        else:
            return self.get_random_canditates(self.n_initial_trailing)

    def get_additional_candidates_trailing(self, best_current_candidate: object) -> List[object]:
        root_expressions = self.search_space.get_root_expressions()
        additional_candidates = []
        # Add random walks from root (exploration)
        for _ in range(0, self.n_exploration_trailing):
            initial_expression = np.random.choice(root_expressions)
            length = np.random.geometric(self.exploration_p_geometric)
            additional_candidates += self.search_space.random_walk(length, initial_expression)
        # Add candidates around current best - either random walks with a specified walk length or all direct neighbours (exploitation)
        if self.do_random_walk_exploitation_trailing:
            n_walks = int(self.n_exploitation_trailing / self.walk_length_exploitation_trailing)
            for _ in range(0, n_walks):
                additional_candidates += self.search_space.random_walk(
                    self.walk_length_exploitation_trailing, best_current_candidate
                )
        else:
            around_best = self.search_space.get_neighbour_expressions(best_current_candidate)
            if self.limit_n_additional_current_best and len(around_best) > self.n_exploitation_trailing:
                around_best = list(np.random.choice(around_best, self.n_exploitation_trailing, replace=False))
            additional_candidates += around_best
        return additional_candidates

    def get_random_candidate_n_operations(self, n: int):
        root_expressions = self.search_space.get_root_expressions()
        initial_expression = np.random.choice(root_expressions)
        candidate = self.search_space.random_walk(n, initial_expression)[-1]
        return candidate

    def get_around_candidate_for_evolutionary_opt(self, candidate: object, n_around_candidate: int):
        """
        Generates random walks in the search space from a given candidate - used in object evolutionary algorithm
        """
        expression_list = []
        for i in range(0, n_around_candidate):
            new_expression = self.search_space.get_random_neighbour_expression(candidate)
            expression_list.append(new_expression)
        return expression_list

    def get_initial_for_evolutionary_opt(self, n_initial):
        return self.get_dataset_recursivly_generated(n_initial, 1)

    def get_dataset_recursivly_generated(self, n_data, n_per_step, filter_out_equivalent_expressions=False):
        expression_list = self.search_space.get_root_expressions()
        # recursivly add n_per_step around a randomly chosen element of the list to the list until n_data is reached
        while len(expression_list) < n_data:
            chosen_expression = np.random.choice(expression_list)
            for i in range(0, n_per_step):
                if len(expression_list) < n_data:
                    new_expression = self.search_space.get_random_neighbour_expression(chosen_expression)
                    if filter_out_equivalent_expressions:
                        while self.check_if_equivalent_expression_in_list(new_expression, expression_list):
                            logger.info("Expression already in list - sample new neighbour")
                            new_expression = self.search_space.get_random_neighbour_expression(chosen_expression)
                    expression_list.append(new_expression)
        return expression_list

    def check_if_equivalent_expression_in_list(self, expression, expression_list):
        for expression_in_list in expression_list:
            if self.search_space.check_expression_equality(expression_in_list, expression):
                return True
        return False

    def get_input_dimension(self):
        return self.search_space.input_dimension


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    search_space = CompositionalKernelSearchSpace(1)
    generator = KernelGrammarCandidateGenerator(search_space, 3, 10, 0.25, 10, 2, True, False)
    for expression in generator.get_initial_candidates_trailing():
        print(expression)
