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

from typing import Dict, List, Optional, Tuple, Union
import gpflow
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import logging
from alef.configs.kernels.change_hyperplane_kernel_config import BasicCHConfig
from copy import deepcopy
from alef.kernels.base_elementary_kernel import BaseElementaryKernel
from alef.kernels.change_hyperplane_kernel import ChangeHyperplaneKernel
from alef.kernels.pytorch_kernels.elementary_kernels_pytorch import BaseElementaryKernelPytorch

logger = logging.getLogger(__name__)


class KernelGrammarOperator(Enum):
    ADD = 1
    MULTIPLY = 2
    SPLIT_CH = 3


class BaseKernelsLibrary(Enum):
    GPFLOW = 1
    GPYTORCH = 2


ElementaryKernel = Union[BaseElementaryKernel, BaseElementaryKernelPytorch]


class KernelGrammarOperatorWrapper:
    """
    A way to wrap an operator enum into an object to add additinal inforamation. Mainly used for the CP operator
    as it needs the dimension at which it is applied. The wrapper acts as an KernelGrammarOperator object in the
    sense that comparisons can be made against KernelGrammarOperator objects.
    """

    def __init__(self, operator: KernelGrammarOperator) -> None:
        assert isinstance(operator, KernelGrammarOperator)
        self.operator = operator
        self.dimension = None

    def set_dimension(self, dimension: int):
        """
        Sets the dimension at which the operator is applied.

        Args:
            dimension (int): The dimension to set.
        """
        self.dimension = dimension

    def get_dimension(self):
        """
        Gets the dimension at which the operator is applied.

        Returns:
            int: The dimension.
        """
        return self.dimension

    def __eq__(self, operator: object) -> bool:
        if isinstance(operator, KernelGrammarOperator):
            return self.operator == operator
        elif isinstance(operator, self.__class__):
            return self.operator == operator.operator

    def __ne__(self, operator: object) -> bool:
        return not self.__eq__(operator)


class BaseKernelGrammarExpression(ABC):
    """
    Base Object that represents a symbolic expression that corresponds to a kernel. It either represents a basic kernel (ElementaryKernelGrammarExpression)
    or a binary tree consisting of operator suchs as ADD and MULTIPLY as its nodes and base kernels as its children (KernelGrammarExpression). An instance of this class
    always correconds to a gpflow.kernel.Kernel instance which can be resolved via the get_kernel() method.
    """

    @abstractmethod
    def get_kernel(self) -> ElementaryKernel:
        """
        Method to resolve/get the corresponding kernel object
        Return
            gpflow.kernels.Kernel - returns the associated kernel object
        """
        raise NotImplementedError

    @abstractmethod
    def deep_copy(self):
        """
        Returns a deep copy of itself - creates copies of all base kernels and subexpressions
        - thus if weight sharing was active between kernels in subexpression this is not the case for the returned object
        Return:
         BaseKernelGrammarExpression - a deep copy of its self
        """
        raise NotImplementedError

    @abstractmethod
    def count_elementary_expressions(self) -> int:
        """
        Counts number of elementary expressions (instances of ElementaryKernelGrammarExpression) inside the expression
        """
        raise NotImplementedError

    @abstractmethod
    def count_operators(self):
        """
        Counts the number of operators in the binary tree (only greater than 0 for instances of KernelGrammarExpression)
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self):
        """
        Returns a unique name based on the tree structure (operators) and the base kernels in the leaves
        """
        raise NotImplementedError

    @abstractmethod
    def get_hash(self) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        Calcuates hash of the expression based on hashes of all containing subexpressions -> makes it possible to count number
        of identical subexpressions - is used by the get_subtree_dict method
        Return:
         int - hash value of expression - hash is generated recursivly and is invariant to rotation of the nodes in the binary tree
         List[(int,int,int)] - list over all subexpressions with tuples containing: (hash values of subexpressions,number of elementary expressions inside subexpression,depth of the root of the subexpression)
        """
        raise NotImplementedError

    @abstractmethod
    def get_add_mult_invariant_hash(self) -> Tuple[int, List[Tuple[int, int, int]]]:
        raise NotImplementedError

    @abstractmethod
    def get_subtree_dict(self) -> Dict[int, List]:
        """
        Generates dict where each key is a hash of a subexpression and the value is a List containing the number how often the subexpression appeared inside this expression at index 0 -
        this method allows for example the tree grammar kernel kernel to count how often the same subexpression appeared in two different expressions - at index 1 a SubtreeMetaInformation
        object is stored associated to the subexpression
        """
        raise NotImplementedError

    @abstractmethod
    def get_subtree_dict_without_leafs(self) -> Dict[int, List]:
        raise NotImplementedError

    @abstractmethod
    def get_add_mult_invariant_subtree_dict(self) -> Dict[int, List]:
        raise NotImplementedError

    @abstractmethod
    def get_input_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_operator(self) -> Optional[KernelGrammarOperator]:
        raise NotImplementedError

    @abstractmethod
    def push_down_one_mult(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_reduced_paths_to_elementaries(self, operators_to_reduce=[], use_name_as_key=False):
        raise NotImplementedError

    @abstractmethod
    def get_elementary_path_dict(self, operators_to_reduce=[]) -> Dict[int, List]:
        raise NotImplementedError

    @abstractmethod
    def get_elementary_count_dict(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_contains_elementary_dict(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_elementary_below_operator_dict(self, operator: KernelGrammarOperator):
        raise NotImplementedError

    @abstractmethod
    def render_pickable(self):
        raise NotImplementedError

    @abstractmethod
    def set_generator_name(self, generator_name: str):
        """
        In case the grammar expression was generated by a kernel grammar generator the name of the generator could
        be set here - this allows downstream algorithms information of the possible elementary expressions and operators
        that can be present in the expression
        """
        raise NotImplementedError

    @abstractmethod
    def get_generator_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_indexes_of_subexpression(self) -> List[List[int]]:
        """Returns lists of the form [0,0,1,0,...] for each subexpression
        which indexes each subexpression with its own list -> it specifies the path in the tree to the subexpression
        [-1] is a special index list referencing to self/root - expressions corresponding to the index_list can be retrieved by
        get_expression_at_index
        """
        raise NotImplementedError

    @abstractmethod
    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_expression_at_index(self, index_list: List[int]):
        """
        Returns subexpression at index_list where index_list has the form [0,1,1,...] specifying the way down the tree
        0 go down expression1, 1 go down expression2. [-1] is the index_list for the expression itself
        """
        raise NotImplementedError

    @abstractmethod
    def get_base_kernel_library(self) -> BaseKernelsLibrary:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_name()


class SubtreeMetaInformation:
    def __init__(self, hash_value: int, num_elementary: int, depth_counter: int):
        self.hash_value = hash_value
        self.num_elementary = num_elementary
        self.depth_counter = depth_counter
        self.upstream_operators = []

    def increase_depth_count(self):
        """
        Increases the depth counter by one.
        """
        self.depth_counter += 1

    def add_upstream_operator(self, operator: KernelGrammarOperator):
        """
        Adds an upstream operator to the list.

        Args:
            operator (KernelGrammarOperator): The operator to add.
        """
        self.upstream_operators.append(operator)

    def __str__(self):
        return (
            "[Hash: "
            + str(self.hash_value)
            + " num_elementary: "
            + str(self.num_elementary)
            + " depth_counter: "
            + str(self.depth_counter)
            + " upstream_operators: "
            + str(self.upstream_operators)
            + "]"
        )


class ElementarySubtreeMetaInformation(SubtreeMetaInformation):
    def __init__(self, hash_value: int, num_elementary: int, depth_counter: int, elementary_name: str):
        super().__init__(hash_value, num_elementary, depth_counter)
        self.elementary_name = elementary_name


class ElementaryPathMetaInformation:
    def __init__(self, elementary_hash: int, upstream_operator_list: List[KernelGrammarOperator]):
        self.elementary_hash = elementary_hash
        self.upstream_operator_list = upstream_operator_list
        self.hash = hash(str(self.upstream_operator_list + [self.elementary_hash]))

    def generate_operator_count_dict(self):
        """
        Generates a dictionary counting the occurrences of each operator in the upstream operator list.

        Returns:
            dict: A dictionary with operators as keys and their counts as values.
        """
        operator_count_dict = {}
        for operator in self.upstream_operator_list:
            if operator in operator_count_dict:
                operator_count_dict[operator] += 1
            else:
                operator_count_dict[operator] = 1
        return operator_count_dict

    def get_hash(self):
        """
        Gets the hash of the elementary path meta information.

        Returns:
            int: The hash value.
        """
        return self.hash


class ElementaryKernelGrammarExpression(BaseKernelGrammarExpression):
    def __init__(self, kernel: ElementaryKernel):
        self.check_kernel_validity(kernel)
        self.kernel = kernel
        self.generator_name = None
        self.base_kernel_library = self.extract_kernel_library(kernel)

    def set_kernel(self, kernel: ElementaryKernel):
        self.check_kernel_validity(kernel)
        kernel_library = self.extract_kernel_library(kernel)
        assert self.base_kernel_library == kernel_library, "Kernel library can not be changed after object creation"
        self.kernel = kernel

    def check_kernel_validity(self, kernel: ElementaryKernel):
        assert isinstance(kernel, BaseElementaryKernel) or isinstance(kernel, BaseElementaryKernelPytorch)

    def extract_kernel_library(self, kernel: ElementaryKernel):
        if isinstance(kernel, BaseElementaryKernel):
            return BaseKernelsLibrary.GPFLOW
        elif isinstance(kernel, BaseElementaryKernelPytorch):
            return BaseKernelsLibrary.GPYTORCH
        else:
            assert False

    def deep_copy(self):
        if isinstance(self.kernel, BaseElementaryKernel):
            new_kernel = gpflow.utilities.deepcopy(self.kernel)
            assert isinstance(new_kernel, BaseElementaryKernel)
            return ElementaryKernelGrammarExpression(new_kernel)
        elif isinstance(self.kernel, BaseElementaryKernelPytorch):
            new_kernel = deepcopy(self.kernel)
            assert isinstance(new_kernel, BaseElementaryKernelPytorch)
            return ElementaryKernelGrammarExpression(new_kernel)

    def count_elementary_expressions(self):
        return 1

    def count_operators(self):
        return 0

    def get_input_dimension(self):
        return self.kernel.get_input_dimension()

    def get_name(self):
        return self.kernel.name

    def render_pickable(self):
        if isinstance(self.kernel, BaseElementaryKernel):
            gpflow.utilities.reset_cache_bijectors(self.kernel)

    def get_operator(self) -> Optional[KernelGrammarOperator]:
        return None

    def get_hash(self):
        hash_value = hash(self.kernel.name)
        subtree_meta_info = ElementarySubtreeMetaInformation(hash_value, 1, 0, self.get_name())
        return hash_value, [subtree_meta_info]

    def get_add_mult_invariant_hash(self):
        return self.get_hash()

    def get_kernel(self):
        return self.kernel

    def get_indexes_of_subexpression_internal(self) -> List[List[int]]:
        return [[None]]

    def get_indexes_of_subexpression(self) -> List[List[int]]:
        return [[-1]]

    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        return [[-1]]

    def get_expression_at_index(self, index_list) -> BaseKernelGrammarExpression:
        assert len(index_list) == 1
        assert index_list[0] == -1
        return self

    def get_subtree_dict(self):
        subtree_meta_info = self.get_hash()[1][0]
        subtree_dict = {}
        subtree_dict[self.get_hash()[0]] = [1, subtree_meta_info]
        return subtree_dict

    def get_subtree_dict_without_leafs(self) -> Dict[int, List]:
        return self.get_subtree_dict()

    def get_add_mult_invariant_subtree_dict(self):
        subtree_meta_info = self.get_add_mult_invariant_hash()[1][0]
        subtree_dict = {}
        subtree_dict[self.get_add_mult_invariant_hash()[0]] = [1, subtree_meta_info]
        return subtree_dict

    def get_reduced_paths_to_elementaries(self, operators_to_reduce=[], use_name_as_key=False):
        if use_name_as_key:
            return {self.get_name(): [[]]}
        else:
            return {self.get_hash()[0]: [[]]}

    def get_elementary_path_dict(self, operators_to_reduce=[]):
        path_meta_info_object = ElementaryPathMetaInformation(self.get_hash()[0], [])
        path_hash = path_meta_info_object.get_hash()
        final_dict = {}
        final_dict[path_hash] = [1, path_meta_info_object]
        return final_dict

    def get_elementary_count_dict(self) -> Dict:
        return {self.get_name(): 1}

    def get_contains_elementary_dict(self) -> Dict:
        return self.get_elementary_count_dict()

    def get_elementary_below_operator_dict(self, operator: KernelGrammarOperator):
        return {self.get_name(): 0}

    def push_down_one_mult(self) -> bool:
        return False

    def set_generator_name(self, generator_name: str):
        self.generator_name = generator_name

    def get_generator_name(self) -> str:
        return self.generator_name

    def get_base_kernel_library(self) -> BaseKernelsLibrary:
        return self.base_kernel_library


class KernelGrammarExpression(BaseKernelGrammarExpression):
    def __init__(
        self,
        expression1: BaseKernelGrammarExpression,
        expression2: BaseKernelGrammarExpression,
        operator: KernelGrammarOperator,
    ):
        self.expression1: BaseKernelGrammarExpression = expression1
        self.expression2: BaseKernelGrammarExpression = expression2
        assert self.expression1.get_base_kernel_library() == self.expression2.get_base_kernel_library(), (
            "Fusion of kernels of different base libraries not possible"
        )
        self.base_kernel_library = self.expression1.get_base_kernel_library()
        self.operator: KernelGrammarOperator = operator
        self.generator_name = None

    def get_kernel(self):
        if self.operator == KernelGrammarOperator.ADD:
            return self.expression1.get_kernel() + self.expression2.get_kernel()
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            return self.expression1.get_kernel() * self.expression2.get_kernel()
        elif self.operator == KernelGrammarOperator.SPLIT_CH:
            assert self.base_kernel_library == BaseKernelsLibrary.GPFLOW, (
                "CH operator not yet implemented for Gpytorch kernels"
            )
            ch_kernel_config = BasicCHConfig(input_dimension=self.get_input_dimension())
            ch_kernel = ChangeHyperplaneKernel(
                self.expression1.get_kernel(), self.expression2.get_kernel(), **ch_kernel_config.dict()
            )
            return ch_kernel

    def get_operator(self) -> Optional[KernelGrammarOperator]:
        return self.operator

    def deep_copy(self):
        new_expression_1 = self.expression1.deep_copy()
        new_expression_2 = self.expression2.deep_copy()
        new_expression = KernelGrammarExpression(
            expression1=new_expression_1, expression2=new_expression_2, operator=self.operator
        )
        return new_expression

    def get_left_expression(self):
        """
        Gets the left subexpression.

        Returns:
            BaseKernelGrammarExpression: The left subexpression.
        """
        return self.expression1

    def get_right_expression(self):
        """
        Gets the right subexpression.

        Returns:
            BaseKernelGrammarExpression: The right subexpression.
        """
        return self.expression2

    def get_input_dimension(self):
        assert self.expression1.get_input_dimension() == self.expression2.get_input_dimension()
        return self.expression1.get_input_dimension()

    def render_pickable(self):
        self.expression1.render_pickable()
        self.expression2.render_pickable()

    def count_elementary_expressions(self):
        return self.expression1.count_elementary_expressions() + self.expression2.count_elementary_expressions()

    def count_operators(self):
        return self.expression1.count_operators() + self.expression2.count_operators() + 1

    def get_base_kernel_library(self) -> BaseKernelsLibrary:
        return self.base_kernel_library

    def get_name(self):
        if self.operator == KernelGrammarOperator.ADD:
            return "(" + self.expression1.get_name() + " ADD " + self.expression2.get_name() + ")"
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            return "(" + self.expression1.get_name() + " MULTIPLY " + self.expression2.get_name() + ")"
        elif self.operator == KernelGrammarOperator.SPLIT_CH:
            return "(" + self.expression1.get_name() + " SPLIT_CH " + self.expression2.get_name() + ")"

    def set_generator_name(self, generator_name: str):
        self.generator_name = generator_name

    def get_generator_name(self) -> str:
        return self.generator_name

    def get_indexes_of_subexpression_internal(self) -> List[List[int]]:
        list_left = [[0]]
        list_right = [[1]]
        list_expression_1 = self.expression1.get_indexes_of_subexpression_internal()
        for index_list in list_expression_1:
            extended_index_list = [0] + index_list
            list_left.append(extended_index_list)
        list_expression_2 = self.expression2.get_indexes_of_subexpression_internal()
        for index_list in list_expression_2:
            extended_index_list = [1] + index_list
            list_right.append(extended_index_list)
        return list_left + list_right

    def get_indexes_of_subexpression(self) -> List[List[int]]:
        complete_list = self.get_indexes_of_subexpression_internal()
        reduced_list = [[-1]]
        for index_list in complete_list:
            if index_list[-1] is not None:
                reduced_list.append(index_list)
        return reduced_list

    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        complete_list = self.get_indexes_of_subexpression_internal()
        reduced_list = []
        for index_list in complete_list:
            if index_list[-1] is None:
                reduced_list.append(index_list[:-1])
        return reduced_list

    def get_expression_at_index(self, index_list: List[int]) -> BaseKernelGrammarExpression:
        """
        Returns subexpression at index_list where index_list has the form [0,1,1,...] specifying the way down the tree
        0 go down expression1, 1 go down expression2. [-1] is the index_list for the expression itself

        Args:
            index_list (List[int]): The index list specifying the path to the subexpression.

        Returns:
            BaseKernelGrammarExpression: The subexpression at the specified index.
        """
        if len(index_list) == 1:
            if index_list[0] == 0:
                return self.expression1
            elif index_list[0] == 1:
                return self.expression2
            elif index_list[0] == -1:
                return self
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                return self.expression1.get_expression_at_index(index_list[1:])
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                return self.expression2.get_expression_at_index(index_list[1:])

    def set_expression_at_index(self, index_list: List[int], expression: BaseKernelGrammarExpression):
        """
        Exchanges subexpression at index_list with the expression in the argument where index_list has the form [0,1,1,...] specifying the way down the tree.
        Changing of itself (index list [-1]) is not allowed.

        Args:
            index_list (List[int]): The index list specifying the path to the subexpression.
            expression (BaseKernelGrammarExpression): The new subexpression to set.
        """
        if len(index_list) == 1:
            if index_list[0] == 0:
                self.expression1 = expression
            elif index_list[0] == 1:
                self.expression2 = expression
            elif index_list[0] == -1:
                assert False
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                return self.expression1.set_expression_at_index(index_list[1:], expression)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                return self.expression2.set_expression_at_index(index_list[1:], expression)

    def get_operator_hash(self):
        """
        Gets the hash value of the operator.

        Returns:
            int: The hash value of the operator.
        """
        if self.operator == KernelGrammarOperator.ADD:
            hash_value = hash("ADD")
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            hash_value = hash("MULTIPLY")
        elif self.operator == KernelGrammarOperator.SPLIT_CH:
            hash_value = hash("SPLIT_CH")
        return hash_value

    def get_hash(self):
        hash_collection = []
        hash_collection.append(self.get_operator_hash())
        hash_expr1, meta_info_list_expr1 = self.expression1.get_hash()
        hash_expr2, meta_info_list_expr2 = self.expression2.get_hash()
        for element in meta_info_list_expr1:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        for element in meta_info_list_expr2:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        hash_collection.append(hash_expr1)
        hash_collection.append(hash_expr2)
        hash_collection.sort()
        hash_value = hash(str(hash_collection))
        num_elementary_expressions = self.count_elementary_expressions()
        subtree_meta_info = SubtreeMetaInformation(hash_value, num_elementary_expressions, 0)
        subtree_meta_info_list = [subtree_meta_info] + meta_info_list_expr1 + meta_info_list_expr2
        return hash_value, subtree_meta_info_list

    def get_add_mult_invariant_hash(self):
        assert self.operator == KernelGrammarOperator.ADD or self.operator == KernelGrammarOperator.MULTIPLY
        hash_expr1, meta_info_list_expr1 = self.expression1.get_add_mult_invariant_hash()
        hash_expr2, meta_info_list_expr2 = self.expression2.get_add_mult_invariant_hash()
        for element in meta_info_list_expr1:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        for element in meta_info_list_expr2:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        if self.operator == KernelGrammarOperator.ADD:
            hash_value = hash_expr1 + hash_expr2
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            hash_value = hash_expr1 * hash_expr2
        self.count_elementary_expressions()
        subtree_meta_info = SubtreeMetaInformation(hash_value, 0, 0)
        subtree_meta_info_list = [subtree_meta_info] + meta_info_list_expr1 + meta_info_list_expr2
        return hash_value, subtree_meta_info_list

    def change_elementary_expression_at_index(self, index_list: List[int], new_kernel: ElementaryKernel):
        """
        Changes the elementary expression at the specified index with a new kernel.

        Args:
            index_list (List[int]): The index list specifying the path to the elementary expression.
            new_kernel (ElementaryKernel): The new kernel to set.
        """
        if len(index_list) == 1:
            if index_list[0] == 0:
                assert isinstance(self.expression1, ElementaryKernelGrammarExpression)
                self.expression1 = ElementaryKernelGrammarExpression(new_kernel)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, ElementaryKernelGrammarExpression)
                self.expression2 = ElementaryKernelGrammarExpression(new_kernel)
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                self.expression1.change_elementary_expression_at_index(index_list[1:], new_kernel)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                self.expression2.change_elementary_expression_at_index(index_list[1:], new_kernel)

    def change_random_elementary_expression(self, new_kernel: ElementaryKernel):
        """
        Changes a random elementary expression with a new kernel.

        Args:
            new_kernel (ElementaryKernel): The new kernel to set.

        Returns:
            float: The probability of choosing the current expression.
        """
        probability = 1.0
        if np.random.randint(2):
            current_expression = self.expression1
        else:
            current_expression = self.expression2
        probability = probability * 0.5

        while not isinstance(current_expression, ElementaryKernelGrammarExpression):
            if np.random.randint(2):
                current_expression = current_expression.expression1
            else:
                current_expression = current_expression.expression2
            probability = probability * 0.5

        assert isinstance(current_expression, ElementaryKernelGrammarExpression)
        current_expression.set_kernel(new_kernel)
        return probability

    def extend_sub_expression_at_index(
        self, index_list: List[int], operator: KernelGrammarOperator, new_kernel: ElementaryKernel
    ):
        """
        Extends the subexpression at the specified index with a new kernel and operator.

        Args:
            index_list (List[int]): The index list specifying the path to the subexpression.
            operator (KernelGrammarOperator): The operator to use for extension.
            new_kernel (ElementaryKernel): The new kernel to extend with.
        """
        if len(index_list) == 1:
            if index_list[0] == 0:
                self.expression1 = KernelGrammarExpression(
                    self.expression1, ElementaryKernelGrammarExpression(new_kernel), operator
                )
            elif index_list[0] == 1:
                self.expression2 = KernelGrammarExpression(
                    self.expression2, ElementaryKernelGrammarExpression(new_kernel), operator
                )
            elif index_list[0] == -1:
                self.expression1 = self.deep_copy()
                self.expression2 = ElementaryKernelGrammarExpression(new_kernel)
                self.operator = operator
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                self.expression1.extend_sub_expression_at_index(index_list[1:], operator, new_kernel)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                self.expression2.extend_sub_expression_at_index(index_list[1:], operator, new_kernel)

    def extend_random_sub_expression(
        self,
        operator: KernelGrammarOperator,
        new_kernel: ElementaryKernel,
        skip_operator=False,
        operator_to_skip=KernelGrammarOperator.SPLIT_CH,
    ):
        """
        Extends a random subexpression with a new kernel and operator.

        Args:
            operator (KernelGrammarOperator): The operator to use for extension.
            new_kernel (ElementaryKernel): The new kernel to extend with.
            skip_operator (bool, optional): Whether to skip certain operators. Defaults to False.
            operator_to_skip (KernelGrammarOperator, optional): The operator to skip. Defaults to KernelGrammarOperator.SPLIT_CH.
        """
        choose_first = np.random.randint(2)
        if choose_first:
            chosen_sub_expression = self.expression1
        else:
            chosen_sub_expression = self.expression2

        if isinstance(chosen_sub_expression, ElementaryKernelGrammarExpression):
            progress = False
        elif skip_operator and chosen_sub_expression.get_operator() == operator_to_skip:
            progress = True
        else:
            progress = np.random.randint(2)

        if progress:
            assert isinstance(chosen_sub_expression, KernelGrammarExpression)
            chosen_sub_expression.extend_random_sub_expression(operator, new_kernel)
        else:
            if choose_first:
                self.expression1 = KernelGrammarExpression(
                    self.expression1, ElementaryKernelGrammarExpression(new_kernel), operator
                )
            else:
                self.expression2 = KernelGrammarExpression(
                    self.expression2, ElementaryKernelGrammarExpression(new_kernel), operator
                )
            return

    def extend_top_level_operator(self, operator: KernelGrammarOperator, new_kernel: ElementaryKernel):
        """
        Extends the top-level operator with a new kernel and operator.

        Args:
            operator (KernelGrammarOperator): The operator to use for extension.
            new_kernel (ElementaryKernel): The new kernel to extend with.
        """
        if self.operator == operator:
            choose_first = np.random.randint(2)
            if choose_first:
                chosen_sub_expression = self.expression1
            else:
                chosen_sub_expression = self.expression2

            if isinstance(chosen_sub_expression, ElementaryKernelGrammarExpression):
                progress = False
            elif not chosen_sub_expression.get_operator() == operator:
                progress = False
            else:
                progress = np.random.randint(2)

            if progress:
                assert isinstance(chosen_sub_expression, KernelGrammarExpression)
                chosen_sub_expression.extend_top_level_operator(operator, new_kernel)
            else:
                if choose_first:
                    self.expression1 = KernelGrammarExpression(
                        self.expression1, ElementaryKernelGrammarExpression(new_kernel), operator
                    )
                else:
                    self.expression2 = KernelGrammarExpression(
                        self.expression2, ElementaryKernelGrammarExpression(new_kernel), operator
                    )
                return
        else:
            self.expression1 = KernelGrammarExpression(self.expression1, self.expression2, self.operator)
            self.expression2 = ElementaryKernelGrammarExpression(new_kernel)
            self.operator = operator

    def get_subtree_dict(self):
        _, subtree_meta_info_list = self.get_hash()
        subtree_dict = {}
        for subtree_meta_info in subtree_meta_info_list:
            hash_value = subtree_meta_info.hash_value
            # num_elementary = subtree_meta_info.num_elementary
            if hash_value in subtree_dict:
                subtree_dict[hash_value][0] += 1
            else:
                subtree_dict[hash_value] = [1, subtree_meta_info]

        return subtree_dict

    def get_subtree_dict_without_leafs(self):
        subtree_dict = self.get_subtree_dict()
        subtree_dict_without_leafs = {}
        for hash_value in subtree_dict:
            subtree_meta_info = subtree_dict[hash_value][1]
            assert isinstance(subtree_meta_info, SubtreeMetaInformation)
            if subtree_meta_info.num_elementary > 1:
                subtree_dict_without_leafs[hash_value] = subtree_dict[hash_value]
        return subtree_dict_without_leafs

    def get_add_mult_invariant_subtree_dict(self):
        _, subtree_meta_info_list = self.get_add_mult_invariant_hash()
        subtree_dict = {}
        for subtree_meta_info in subtree_meta_info_list:
            hash_value = subtree_meta_info.hash_value
            # num_elementary = subtree_meta_info.num_elementary
            if hash_value in subtree_dict:
                subtree_dict[hash_value][0] += 1
            else:
                subtree_dict[hash_value] = [1, subtree_meta_info]
        return subtree_dict

    def get_paths_to_elementaries(self, use_name_as_key=False):
        """
        Gets the paths to elementary expressions.

        Args:
            use_name_as_key (bool, optional): Whether to use the name as the key. Defaults to False.

        Returns:
            dict: A dictionary with paths to elementary expressions.
        """
        _, subtree_meta_info_list = self.get_hash()
        paths = {}
        for subtree_meta_info in subtree_meta_info_list:
            hash_value = subtree_meta_info.hash_value
            num_elementary = subtree_meta_info.num_elementary
            if num_elementary == 1 and use_name_as_key:
                key = subtree_meta_info.elementary_name
            else:
                key = hash_value
            if key in paths:
                paths[key].append(subtree_meta_info.upstream_operators)
            elif num_elementary == 1:
                paths[key] = [subtree_meta_info.upstream_operators]
        return paths

    def get_reduced_paths_to_elementaries(self, operators_to_reduce=[], use_name_as_key=False):
        full_paths = self.get_paths_to_elementaries(use_name_as_key)
        reduced_paths = {}
        for elementary_hash in full_paths:
            reduced_paths[elementary_hash] = []
            for upstream_operator_list in full_paths[elementary_hash]:
                reduced_paths[elementary_hash].append(
                    self.reduce_operator_list(upstream_operator_list, operators_to_reduce)
                )
        return reduced_paths

    def reduce_operator_list(self, operator_list, operators_to_reduce):
        """
        Reduces the operator list by removing consecutive duplicate operators.

        Args:
            operator_list (list): The list of operators.
            operators_to_reduce (list): The list of operators to reduce.

        Returns:
            list: The reduced list of operators.
        """
        reduced_list = []
        previous_operator = None
        for operator in operator_list:
            if operator == previous_operator and operator in operators_to_reduce:
                pass
            else:
                reduced_list.append(operator)
            previous_operator = operator
        return reduced_list

    def get_elementary_count_dict(self) -> Dict:
        elementary_count_dict = {}
        elementary_count_dict_1 = self.expression1.get_elementary_count_dict()
        elementary_count_dict_2 = self.expression2.get_elementary_count_dict()
        for key in elementary_count_dict_1:
            elementary_count_dict[key] = elementary_count_dict_1[key]
        for key in elementary_count_dict_2:
            if key in elementary_count_dict:
                elementary_count_dict[key] += elementary_count_dict_2[key]
            else:
                elementary_count_dict[key] = elementary_count_dict_2[key]
        return elementary_count_dict

    def get_contains_elementary_dict(self) -> Dict:
        elementary_count_dict = self.get_elementary_count_dict()
        contains_elementary_dict = {}
        for key in elementary_count_dict:
            if elementary_count_dict[key] > 0:
                contains_elementary_dict[key] = 1
            else:
                contains_elementary_dict[key] = 0
        return contains_elementary_dict

    def get_elementary_path_dict(self, operators_to_reduce=[]):
        # Counts and indexes elementary paths e.g. ADD-> ADD-> MULTIPLY-> LINEAR
        # each path has a unique hash that is the key in this dict
        reduced_paths = self.get_reduced_paths_to_elementaries(operators_to_reduce)
        final_dict = {}
        for elementary_hash in reduced_paths:
            for path in reduced_paths[elementary_hash]:
                path_meta_info_object = ElementaryPathMetaInformation(elementary_hash, path)
                path_hash = path_meta_info_object.get_hash()
                if path_hash in final_dict:
                    final_dict[path_hash][0] += 1
                else:
                    final_dict[path_hash] = [1, path_meta_info_object]
        return final_dict

    def get_elementary_below_operator_dict(self, operator: KernelGrammarOperator):
        paths = self.get_paths_to_elementaries(True)
        final_dict = {}
        for elementary_name in paths:
            final_dict[elementary_name] = 0
            for upstream_operator_list in paths[elementary_name]:
                if operator in upstream_operator_list:
                    final_dict[elementary_name] += 1
        return final_dict

    def push_down_one_mult(self):
        """
        Pushes down multiplication and addition - does change the form of the expression! - does not change the resulting kernel
        Should only be used via the tansform to normal form method in the KernelGrammarExpressionTransformer
        """
        assert self.operator == KernelGrammarOperator.MULTIPLY or self.operator == KernelGrammarOperator.ADD
        if self.operator == KernelGrammarOperator.MULTIPLY:
            if self.get_left_expression().get_operator() == KernelGrammarOperator.ADD:
                self.operator = KernelGrammarOperator.ADD
                expression_right = self.get_right_expression()
                left_expression_left = self.get_left_expression().get_left_expression()
                left_expression_right = self.get_left_expression().get_right_expression()
                self.expression1 = KernelGrammarExpression(
                    left_expression_left, expression_right, KernelGrammarOperator.MULTIPLY
                )
                self.expression2 = KernelGrammarExpression(
                    left_expression_right, expression_right, KernelGrammarOperator.MULTIPLY
                )
                return True
            elif self.get_right_expression().get_operator() == KernelGrammarOperator.ADD:
                self.operator = KernelGrammarOperator.ADD
                expression_left = self.get_left_expression()
                right_expression_left = self.get_right_expression().get_left_expression()
                right_expression_right = self.get_right_expression().get_right_expression()
                self.expression1 = KernelGrammarExpression(
                    right_expression_left, expression_left, KernelGrammarOperator.MULTIPLY
                )
                self.expression2 = KernelGrammarExpression(
                    right_expression_right, expression_left, KernelGrammarOperator.MULTIPLY
                )
                return True

        pushed_down1 = self.expression1.push_down_one_mult()
        if pushed_down1:
            return True
        pushed_down2 = self.expression2.push_down_one_mult()
        if pushed_down2:
            return True
        return False


class KernelGrammarExpressionTransformer:
    @staticmethod
    def transform_to_normal_form(expression: BaseKernelGrammarExpression) -> BaseKernelGrammarExpression:
        """
        Produces a normal form representation - a sum over products representation - of a kernel grammar expression
        - resulting kernel grammar expression is associated with the same kernel.
        """
        new_expression = expression.deep_copy()
        new_expression.set_generator_name(expression.get_generator_name())
        assert isinstance(new_expression, BaseKernelGrammarExpression)
        mult_changed = new_expression.push_down_one_mult()
        while mult_changed:
            mult_changed = new_expression.push_down_one_mult()
        return new_expression

    @staticmethod
    def multiply_expressions(
        expression_list: List[BaseKernelGrammarExpression], make_deep_copy=True
    ) -> BaseKernelGrammarExpression:
        return KernelGrammarExpressionTransformer.combine_all_expressions_via_operator(
            KernelGrammarOperator.MULTIPLY, expression_list, make_deep_copy
        )

    @staticmethod
    def add_expressions(
        expression_list: List[BaseKernelGrammarExpression], make_deep_copy=True
    ) -> BaseKernelGrammarExpression:
        return KernelGrammarExpressionTransformer.combine_all_expressions_via_operator(
            KernelGrammarOperator.ADD, expression_list, make_deep_copy
        )

    @staticmethod
    def combine_all_expressions_via_operator(
        operator: KernelGrammarOperator, expression_list: List[BaseKernelGrammarExpression], make_deep_copy=True
    ) -> BaseKernelGrammarExpression:
        if len(expression_list) == 1:
            return expression_list[0]
        assert len(expression_list) > 1
        new_expression = expression_list[0]
        for i in range(1, len(expression_list)):
            if i % 2 == 0:
                new_expression = KernelGrammarExpression(new_expression, expression_list[i], operator=operator)
            else:
                new_expression = KernelGrammarExpression(expression_list[i], new_expression, operator=operator)
        if make_deep_copy:
            return new_expression.deep_copy()
        else:
            return new_expression
