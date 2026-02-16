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

from abc import ABC, abstractmethod
import logging
from typing import Tuple, Optional
import gpflow
import numpy as np
from alef.kernels.additive_kernel import Partition
from alef.configs.kernels.additive_kernel_configs import AdditiveKernelWithPriorConfig
from alef.kernels.kernel_factory import KernelFactory
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig
from alef.configs.kernels.linear_configs import BasicLinearConfig, LinearWithPriorConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.kernels.kernel_grammar.kernel_grammar import (
    ElementaryKernelGrammarExpression,
    BaseKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
import scipy
import random
import math

logger = logging.getLogger(__name__)


class BaseGpModelMCMCState(ABC):
    @abstractmethod
    def get_kernel(self) -> gpflow.kernels.Kernel:
        raise NotImplementedError

    @abstractmethod
    def get_prior_probability(self) -> Tuple[bool, Optional[float]]:
        """
        Method to get prior probability of the state in the markov chain

        Returns:
            bool - True if prior is uniform (discrete state space: only allowed if state space is finite)
            Optional(float) - prior probability (will not be considered if upper flag is True and can be set to None in this case)
        """
        raise NotImplementedError


class BaseGpModelMCMCProposal(ABC):
    @abstractmethod
    def propose_next(self) -> Tuple[BaseGpModelMCMCState, np.float, np.float]:
        """
        Method to sample a new Kernel/Model M' from the proposal distribution - current state is M

        Returns:
             gpflow.kernels.Kernel - proposed kernel
             np.float - probability value P(M'|M)
             np.float - probability value P(M|M')
        """
        raise NotImplementedError

    @abstractmethod
    def accept(self):
        """
        Method to indicate that proposed kernel was accepted - internal state should then switch to the new kernel M<-M'
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_state(self) -> BaseGpModelMCMCState:
        """
        Retrieves current kernel/state in proposal M
        """
        raise NotImplementedError

    @abstractmethod
    def set_current_state(self, state: BaseGpModelMCMCState):
        """
        Set the current state of the chain from which proposals are drawn

        Arguments:
            state : BaseGpModelMCMCState - State M from which the proposal draws P(M'|M)
        """
        raise NotImplementedError


class AdditiveGPMCMCState(BaseGpModelMCMCState):
    def __init__(self, partition: Partition):
        partition.check_partition_validity()
        self.partition = partition

    def set_partition(self, partition: Partition):
        partition.check_partition_validity()
        self.partition = partition

    def get_partition(self) -> Partition:
        return self.partition

    def get_prior_probability(self) -> Tuple[bool, Optional[float]]:
        return True, None

    def get_kernel(self) -> gpflow.kernels.Kernel:
        kernel_config = AdditiveKernelWithPriorConfig(
            input_dimension=self.partition.get_num_dims(), partition=self.partition
        )
        return KernelFactory().build(kernel_config)


class AdditiveGPMCMCProposal(BaseGpModelMCMCProposal):
    def __init__(self, initial_state: AdditiveGPMCMCState):
        self.state = initial_state
        self.proposed_state = None

    def propose_next(self) -> Tuple[AdditiveGPMCMCState, np.float, np.float]:
        current_partition = self.state.get_partition()
        partition_list = current_partition.get_partition_list()
        split = np.random.random() < 0.5
        if split:
            new_partition_list, p_m_prop_m_curr, p_m_curr_m_prop = self.split(partition_list)
        else:
            new_partition_list, p_m_prop_m_curr, p_m_curr_m_prop = self.merge(partition_list)
        new_partition = Partition(current_partition.get_num_dims())
        new_partition.set_partition_list(new_partition_list)
        self.proposed_state = AdditiveGPMCMCState(new_partition)
        logger.info("Current partition list:")
        logger.info(partition_list)
        logger.info("Proposed partition list:")
        logger.info(self.proposed_state.get_partition().get_partition_list())
        return self.proposed_state, p_m_prop_m_curr, p_m_curr_m_prop

    def split(self, partition_list):
        logger.info("SPLIT")
        num_partitions = len(partition_list)
        p_m_prop_m_curr = 0.5
        p_m_curr_m_prop = 0.5
        chosen_list_index = np.random.choice(a=num_partitions)
        element_to_be_splitted = partition_list[chosen_list_index]
        element_length = len(element_to_be_splitted)
        if element_length > 1:
            chosen_split_index = np.random.choice(a=element_length - 1)
            element_to_be_splitted_copy = element_to_be_splitted.copy()
            random.shuffle(element_to_be_splitted_copy)
            split_0 = element_to_be_splitted_copy[: (chosen_split_index + 1)]
            split_1 = element_to_be_splitted_copy[(chosen_split_index + 1) :]
            p_m_prop_m_curr = (
                p_m_prop_m_curr
                * (1.0 / float(num_partitions))
                * (1.0 / float(element_length - 1))
                * (1.0 / math.factorial(element_length))
            )
            p_m_curr_m_prop = (
                p_m_curr_m_prop
                * (1.0 / float(scipy.special.binom(num_partitions + 1, 2)))
                * (1.0 / math.factorial(element_length))
            )
            new_partition_list = partition_list.copy()
            new_partition_list.remove(element_to_be_splitted)
            new_partition_list.append(split_0)
            new_partition_list.append(split_1)
        else:
            number_single_dimension_elements = len([element for element in partition_list if len(element) == 1])
            p_m_prop_m_curr = p_m_prop_m_curr * float(number_single_dimension_elements) / float(len(partition_list))
            p_m_curr_m_prop = p_m_prop_m_curr
            new_partition_list = partition_list.copy()
        return new_partition_list, p_m_prop_m_curr, p_m_curr_m_prop

    def merge(self, partition_list):
        logger.info("MERGE")
        num_partitions = len(partition_list)
        p_m_prop_m_curr = 0.5
        p_m_curr_m_prop = 0.5
        if num_partitions > 1:
            chosen_list_indexes = np.random.choice(a=num_partitions, size=2, replace=False)

            element_0 = partition_list[chosen_list_indexes[0]]
            element_1 = partition_list[chosen_list_indexes[1]]
            new_element = element_0 + element_1
            random.shuffle(new_element)
            new_partition_list = partition_list.copy()
            new_partition_list.remove(element_0)
            new_partition_list.remove(element_1)
            new_partition_list.append(new_element)
            p_m_prop_m_curr = (
                p_m_prop_m_curr
                * 1.0
                / float(scipy.special.binom(num_partitions, 2))
                * (1.0 / math.factorial(len(new_element)))
            )
            p_m_curr_m_prop = (
                p_m_curr_m_prop
                * (1.0 / float(len(new_partition_list)))
                * (1.0 / float(len(new_element) - 1))
                * (1.0 / math.factorial(len(new_element)))
            )
        else:
            new_partition_list = partition_list.copy()
            random.shuffle(new_partition_list)
            p_m_prop_m_curr = p_m_prop_m_curr * (1.0 / math.factorial(len(new_partition_list)))
            p_m_curr_m_prop = p_m_curr_m_prop * (1.0 / math.factorial(len(new_partition_list)))

        return new_partition_list, p_m_prop_m_curr, p_m_curr_m_prop

    def accept(self):
        self.state = self.proposed_state

    def get_current_state(self) -> BaseGpModelMCMCState:
        return self.state

    def set_current_state(self, state: BaseGpModelMCMCState):
        self.state = state


class KernelGrammarMCMCState(BaseGpModelMCMCState):
    def __init__(self, expression: BaseKernelGrammarExpression, number_of_base_kernels: int) -> None:
        self.expression = expression
        self.number_of_base_kernels = number_of_base_kernels
        self.number_of_operator_types = 2

    def set_expression(self, expression: BaseKernelGrammarExpression):
        self.expression = expression

    def get_expression(self):
        return self.expression

    def get_prior_probability(self) -> Tuple[bool, Optional[float]]:
        number_elemtary_expressions = self.expression.count_elementary_expressions()
        number_operators = self.expression.count_operators()
        prior_probability = np.power(1.0 / float(self.number_of_base_kernels), number_elemtary_expressions) * np.power(
            1.0 / float(self.number_of_operator_types), number_operators
        )
        return False, prior_probability

    def get_kernel(self) -> gpflow.kernels.Kernel:
        return self.expression.get_kernel()


class KernelGrammarMCMCProposal(BaseGpModelMCMCProposal):
    def __init__(self, initial_kernel_config: BaseKernelConfig, add_hp_prior=True):
        self.input_dimension = initial_kernel_config.input_dimension
        self.add_hp_pior = add_hp_prior
        self.base_kernel_config_list_without_priors = [BasicRBFConfig, BasicMatern52Config, BasicLinearConfig]
        self.base_kernel_config_list_with_priors = [RBFWithPriorConfig, Matern52WithPriorConfig, LinearWithPriorConfig]
        if self.add_hp_pior:
            self.base_kernel_config_list = [
                config(input_dimension=self.input_dimension) for config in self.base_kernel_config_list_with_priors
            ]
        else:
            self.base_kernel_config_list = [
                config(input_dimension=self.input_dimension) for config in self.base_kernel_config_list_without_priors
            ]

        self.num_base_kernels = len(self.base_kernel_config_list)
        initial_expression = ElementaryKernelGrammarExpression(KernelFactory.build(initial_kernel_config))
        self.state = KernelGrammarMCMCState(initial_expression, self.num_base_kernels)
        self.proposed_state = None

    def propose_next(self) -> Tuple[KernelGrammarMCMCState, np.float, np.float]:
        action_index = np.random.randint(4)

        if action_index == 0:
            new_expression, p_m_prop_m_curr, p_m_curr_m_prop = self.add()
        elif action_index == 1:
            new_expression, p_m_prop_m_curr, p_m_curr_m_prop = self.multiply()
        elif action_index == 2:
            new_expression, p_m_prop_m_curr, p_m_curr_m_prop = self.delete()
        elif action_index == 3:
            new_expression, p_m_prop_m_curr, p_m_curr_m_prop = self.change()

        self.proposed_state = KernelGrammarMCMCState(new_expression, self.num_base_kernels)
        logger.info(self.proposed_state.get_expression().get_name())
        logger.debug("Prob given current")
        logger.debug(p_m_prop_m_curr)
        logger.debug("Prob given proposed")
        logger.debug(p_m_curr_m_prop)

        return self.proposed_state, p_m_prop_m_curr, p_m_curr_m_prop

    def add(self):
        logger.info("ADD")
        p_m_prop_m_curr = 0.25
        p_m_curr_m_prop = 0.25
        current_expression = self.state.get_expression()
        chosen_base_kernel_config = random.choice(self.base_kernel_config_list)
        p_m_prop_m_curr = p_m_prop_m_curr * (1.0 / len(self.base_kernel_config_list))
        elementary_expression = ElementaryKernelGrammarExpression(KernelFactory.build(chosen_base_kernel_config))
        new_expression = KernelGrammarExpression(current_expression, elementary_expression, KernelGrammarOperator.ADD)
        return new_expression, p_m_prop_m_curr, p_m_curr_m_prop

    def delete(self):
        logger.info("DELETE")
        p_m_prop_m_curr = 0.25
        p_m_curr_m_prop = 0.25
        current_expression = self.state.get_expression()
        if isinstance(current_expression, KernelGrammarExpression):
            new_expression = current_expression.get_left_expression()
            p_m_curr_m_prop = p_m_curr_m_prop * (1.0 / len(self.base_kernel_config_list))
            return new_expression, p_m_prop_m_curr, p_m_curr_m_prop
        elif isinstance(current_expression, ElementaryKernelGrammarExpression):
            return current_expression, p_m_prop_m_curr, p_m_curr_m_prop

    def multiply(self):
        logger.info("MULTIPLY")
        p_m_prop_m_curr = 0.25
        p_m_curr_m_prop = 0.25
        current_expression = self.state.get_expression()
        chosen_base_kernel_config = random.choice(self.base_kernel_config_list)
        p_m_prop_m_curr = p_m_prop_m_curr * (1.0 / len(self.base_kernel_config_list))
        elementary_expression = ElementaryKernelGrammarExpression(KernelFactory.build(chosen_base_kernel_config))
        new_expression = KernelGrammarExpression(
            current_expression, elementary_expression, KernelGrammarOperator.MULTIPLY
        )
        return new_expression, p_m_prop_m_curr, p_m_curr_m_prop

    def change(self):
        logger.info("CHANGE")
        p_m_prop_m_curr = 0.25
        p_m_curr_m_prop = 0.25
        current_expression = self.state.get_expression()
        chosen_base_kernel_config = random.choice(self.base_kernel_config_list)
        if isinstance(current_expression, ElementaryKernelGrammarExpression):
            new_expression = ElementaryKernelGrammarExpression(KernelFactory.build(chosen_base_kernel_config))
            p_m_prop_m_curr = p_m_prop_m_curr * (1.0 / len(self.base_kernel_config_list))
            p_m_curr_m_prop = p_m_curr_m_prop * (1.0 / len(self.base_kernel_config_list))
        elif isinstance(current_expression, KernelGrammarExpression):
            new_expression = current_expression.deep_copy()
            prob = new_expression.change_random_elementary_expression(KernelFactory.build(chosen_base_kernel_config))
            p_m_prop_m_curr = p_m_prop_m_curr * (1.0 / len(self.base_kernel_config_list)) * prob
            p_m_curr_m_prop = p_m_curr_m_prop * (1.0 / len(self.base_kernel_config_list)) * prob
        return new_expression, p_m_prop_m_curr, p_m_curr_m_prop

    def accept(self):
        self.state = self.proposed_state

    def set_current_state(self, state: KernelGrammarMCMCState):
        self.state = state

    def get_current_state(self) -> KernelGrammarMCMCState:
        return self.state


if __name__ == "__main__":
    base_expression_1 = ElementaryKernelGrammarExpression(gpflow.kernels.RBF())
    base_expression_2 = ElementaryKernelGrammarExpression(gpflow.kernels.Matern32())
    base_expression_3 = ElementaryKernelGrammarExpression(gpflow.kernels.Linear())
    expression = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(expression, base_expression_3, KernelGrammarOperator.MULTIPLY)
    print(expression2.get_name())
    print(expression2.count_elementary_expressions())
    print(expression2.count_operators())
    state1 = KernelGrammarMCMCState(expression, 4)
    state2 = KernelGrammarMCMCState(expression2, 4)
    print(state1.get_prior_probability()[1])
    print(state2.get_prior_probability()[1])
