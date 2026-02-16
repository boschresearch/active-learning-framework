import os
import pickle
import numpy as np
import pytest
import torch
from alef.configs.kernels.kernel_grammar_generators.compositional_kernel_search_configs import (
    CompositionalKernelSearchGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.dynamic_hhk_generator_config import DynamicHHKGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import (
    NDimFullKernelsGrammarGeneratorConfig,
)
from alef.configs.kernels.linear_configs import BasicLinearConfig, LinearWithPriorConfig
from alef.configs.kernels.periodic_configs import BasicPeriodicConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicLinearKernelPytorchConfig,
    BasicPeriodicKernelPytorchConfig,
    BasicRBFPytorchConfig,
)
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.kernels.kernel_factory import KernelFactory
from alef.kernels.kernel_grammar.generator_factory import GeneratorFactory
from alef.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelsLibrary,
    ElementaryKernelGrammarExpression,
    ElementaryPathMetaInformation,
    KernelGrammarExpression,
    KernelGrammarExpressionTransformer,
    KernelGrammarOperator,
    KernelGrammarOperatorWrapper,
)
from alef.kernels.kernel_grammar.kernel_grammar_search_spaces import CKSWithRQSearchSpace
from alef.kernels.linear_kernel import LinearKernel
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from alef.kernels.rbf_kernel import RBFKernel


@pytest.mark.parametrize("library_type", (BaseKernelsLibrary.GPFLOW, BaseKernelsLibrary.GPYTORCH))
def test_kernel_grammar_hashes(library_type):
    if library_type == BaseKernelsLibrary.GPFLOW:
        base_expression_1 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicRBFConfig(input_dimension=2)))
        base_expression_2 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicLinearConfig(input_dimension=2)))
        base_expression_3 = ElementaryKernelGrammarExpression(
            KernelFactory.build(BasicPeriodicConfig(input_dimension=2))
        )
        base_expression_4 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicRBFConfig(input_dimension=2)))
    elif library_type == BaseKernelsLibrary.GPYTORCH:
        base_expression_1 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicRBFPytorchConfig(input_dimension=2))
        )
        base_expression_2 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicLinearKernelPytorchConfig(input_dimension=2))
        )
        base_expression_3 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicPeriodicKernelPytorchConfig(input_dimension=2))
        )
        base_expression_4 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicRBFPytorchConfig(input_dimension=2))
        )
    expression = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(expression, base_expression_3, KernelGrammarOperator.ADD)
    expression22 = KernelGrammarExpression(base_expression_3, expression, KernelGrammarOperator.ADD)
    expression3 = KernelGrammarExpression(base_expression_4, expression2, KernelGrammarOperator.MULTIPLY)
    expression4 = KernelGrammarExpression(expression22, base_expression_4, KernelGrammarOperator.MULTIPLY)
    hash_expression3, meta_info_list_expr3 = expression3.get_hash()
    hash_expression4, meta_info_list_expr4 = expression4.get_hash()
    assert hash_expression3 == hash_expression4
    for i, meta_info_element_expr3 in enumerate(meta_info_list_expr3):
        hash_value = meta_info_element_expr3.hash_value
        is_included = False
        for j, meta_info_element_expr4 in enumerate(meta_info_list_expr4):
            if hash_value == meta_info_element_expr4.hash_value:
                is_included = True
        assert is_included


def test_kernel_grammar_subpath_dict():
    base_expression_1 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicRBFConfig(input_dimension=2)))
    base_expression_2 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicLinearConfig(input_dimension=2)))
    expression1 = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(expression1, base_expression_1, KernelGrammarOperator.ADD)
    expression3 = KernelGrammarExpression(expression2, base_expression_2, KernelGrammarOperator.ADD)
    expression4 = KernelGrammarExpression(expression3, base_expression_1, KernelGrammarOperator.ADD)
    expression5 = KernelGrammarExpression(expression4, expression1, KernelGrammarOperator.MULTIPLY)
    subpath_dict = expression5.get_elementary_path_dict()
    has_a_two = False
    for key in subpath_dict:
        n, subpath_meta_info = subpath_dict[key]
        if n == 2:
            first_key = key
            has_a_two = True
            assert subpath_meta_info.generate_operator_count_dict()[KernelGrammarOperator.ADD] == 1
    assert has_a_two
    subpath_dict = expression5.get_elementary_path_dict([KernelGrammarOperator.ADD])
    has_a_four = False
    for key in subpath_dict:
        n, subpath_meta_info = subpath_dict[key]
        if n == 4:
            second_key = key

            has_a_four = True
            assert subpath_meta_info.generate_operator_count_dict()[KernelGrammarOperator.ADD] == 1
    assert has_a_four
    assert first_key == second_key


@pytest.mark.parametrize("library_type", (BaseKernelsLibrary.GPFLOW, BaseKernelsLibrary.GPYTORCH))
def test_normal_form_transformer(library_type):
    if library_type == BaseKernelsLibrary.GPFLOW:
        base_expression_1 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicRBFConfig(input_dimension=2)))
        base_expression_2 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicLinearConfig(input_dimension=2)))
        base_expression_3 = ElementaryKernelGrammarExpression(
            KernelFactory.build(BasicPeriodicConfig(input_dimension=2))
        )
        base_expression_4 = ElementaryKernelGrammarExpression(KernelFactory.build(BasicRBFConfig(input_dimension=2)))
    elif library_type == BaseKernelsLibrary.GPYTORCH:
        base_expression_1 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicRBFPytorchConfig(input_dimension=2))
        )
        base_expression_2 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicLinearKernelPytorchConfig(input_dimension=2))
        )
        base_expression_3 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicPeriodicKernelPytorchConfig(input_dimension=2))
        )
        base_expression_4 = ElementaryKernelGrammarExpression(
            PytorchKernelFactory.build(BasicRBFPytorchConfig(input_dimension=2))
        )
    expression = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(expression, base_expression_3, KernelGrammarOperator.ADD)
    expression3 = KernelGrammarExpression(base_expression_4, expression2, KernelGrammarOperator.MULTIPLY)
    expression4 = KernelGrammarExpression(expression2, expression3, KernelGrammarOperator.MULTIPLY)
    expression4_nf = KernelGrammarExpressionTransformer.transform_to_normal_form(expression4)
    kernel4 = expression4.get_kernel()
    kernel4_nf = expression4_nf.get_kernel()
    X = np.array([[1.0, 0.5], [0.1, 0.2], [0.0, 1.0]])
    if library_type == BaseKernelsLibrary.GPFLOW:
        kernel4_evaluated = kernel4.K(X).numpy()
        kernel4_nf_eval = kernel4_nf.K(X).numpy()
    elif library_type == BaseKernelsLibrary.GPYTORCH:
        kernel4_evaluated = kernel4(torch.tensor(X)).numpy()
        kernel4_nf_eval = kernel4_nf(torch.tensor(X)).numpy()
    assert np.allclose(kernel4_evaluated, kernel4_nf_eval)
    elem_dict = expression4_nf.get_elementary_path_dict()
    for key in elem_dict:
        meta_info = elem_dict[key][1]
        assert isinstance(meta_info, ElementaryPathMetaInformation)
        assert str(meta_info.upstream_operator_list) == str(
            sorted(meta_info.upstream_operator_list, key=lambda kernel_operator: kernel_operator.value, reverse=True)
        )
    expression2_nf = KernelGrammarExpressionTransformer.transform_to_normal_form(expression2)
    assert expression2.get_hash()[0] == expression2_nf.get_hash()[0]


@pytest.mark.parametrize(
    "generator_config_class",
    (NDimFullKernelsGrammarGeneratorConfig, CompositionalKernelSearchGeneratorConfig, DynamicHHKGeneratorConfig),
)
def test_kernel_grammar_generators(generator_config_class):
    generator_config = generator_config_class(input_dimension=2)
    generator_config.do_flat_initial_trailing = False
    generator = GeneratorFactory.build(generator_config)
    kernel_expression_list = generator.get_initial_candidates_trailing()
    assert (
        len(kernel_expression_list)
        == generator_config.n_initial_factor_trailing * generator.search_space.get_num_base_kernels()
    )
    for expression in kernel_expression_list:
        assert expression.get_generator_name() == generator.search_space.name
    generator.get_additional_candidates_trailing(kernel_expression_list[0])
    n = generator.search_space.get_num_base_kernels() * 4
    kernel_expression_list = generator.get_random_canditates(n)
    assert len(kernel_expression_list) == n
    for expression in kernel_expression_list:
        assert expression.get_generator_name() == generator.search_space.name


def test_kernel_grammar_indexing():
    base_expression_1 = ElementaryKernelGrammarExpression(RBFKernel(**BasicRBFConfig(input_dimension=2).dict()))
    base_expression_2 = ElementaryKernelGrammarExpression(
        LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict())
    )
    base_expression_3 = ElementaryKernelGrammarExpression(
        LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict())
    )
    base_expression_4 = ElementaryKernelGrammarExpression(RBFKernel(**BasicRBFConfig(input_dimension=2).dict()))
    expression1 = KernelGrammarExpression(base_expression_1, base_expression_2, operator=KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(base_expression_3, base_expression_4, operator=KernelGrammarOperator.SPLIT_CH)
    expression3 = KernelGrammarExpression(expression1, expression2, operator=KernelGrammarOperator.SPLIT_CH)
    expression4 = KernelGrammarExpression(expression3, expression2, operator=KernelGrammarOperator.SPLIT_CH)
    index_base_1_in_4 = [0, 0, 0]
    index_base_3_in_3 = [1, 0]
    assert str(expression2) == str(expression2.get_expression_at_index([-1]))
    assert str(base_expression_1) == str(expression4.get_expression_at_index(index_base_1_in_4))
    assert str(base_expression_3) == str(expression3.get_expression_at_index(index_base_3_in_3))
    all_expressions = [
        str(base_expression_1),
        str(base_expression_2),
        str(base_expression_3),
        str(base_expression_4),
        str(expression1),
        str(expression2),
        str(expression3),
        str(expression4),
    ]
    for sub_expression_index_list in expression4.get_indexes_of_subexpression():
        sub_expression = expression4.get_expression_at_index(sub_expression_index_list)
        assert str(sub_expression) in all_expressions
    all_base_expressions = [base_expression_1, base_expression_2, base_expression_3, base_expression_4]
    for base_expression in all_base_expressions:
        list_index_list = base_expression.get_indexes_of_subexpression()
        assert len(list_index_list) == 1
        assert str(base_expression) == str(base_expression.get_expression_at_index(list_index_list[0]))


def test_pickleing(tmp_path):
    test_path = tmp_path / "test"
    test_path.mkdir()
    base_expression_1 = ElementaryKernelGrammarExpression(RBFKernel(**RBFWithPriorConfig(input_dimension=1).dict()))
    base_expression_2 = ElementaryKernelGrammarExpression(
        LinearKernel(**LinearWithPriorConfig(input_dimension=1).dict())
    )
    base_expression_3 = ElementaryKernelGrammarExpression(
        LinearKernel(**LinearWithPriorConfig(input_dimension=1).dict())
    )
    base_expression_4 = ElementaryKernelGrammarExpression(RBFKernel(**RBFWithPriorConfig(input_dimension=1).dict()))
    expression1 = KernelGrammarExpression(base_expression_1, base_expression_2, operator=KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(base_expression_3, base_expression_4, operator=KernelGrammarOperator.MULTIPLY)
    expression3 = KernelGrammarExpression(expression1, expression2, operator=KernelGrammarOperator.MULTIPLY)
    expression4 = KernelGrammarExpression(expression3, expression2, operator=KernelGrammarOperator.MULTIPLY)
    kernel = expression4.get_kernel()
    kernel.trainable_parameters[0].assign(kernel.trainable_parameters[0].prior.sample())
    expression4 = expression4.deep_copy()
    pickle.dump(expression4, open(os.path.join(test_path, "kernel.p"), "wb"))
    expression4_loaded = pickle.load(open(os.path.join(test_path, "kernel.p"), "rb"))
    assert str(expression4) == str(expression4_loaded)
    X = np.array([[0.0], [0.5], [1.0], [2.0]])
    assert np.allclose(kernel.K(X), expression4_loaded.get_kernel().K(X))


def test_kernel_seach_space():
    search_space = CKSWithRQSearchSpace(3)
    kernel_config_1 = RBFWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1)
    kernel_config_2 = LinearWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=2)
    base_expression_1 = ElementaryKernelGrammarExpression(KernelFactory.build(kernel_config_1))
    base_expression_2 = ElementaryKernelGrammarExpression(KernelFactory.build(kernel_config_2))
    neighbours = search_space.get_neighbour_expressions(base_expression_1)
    neighbour_strings = [str(neighbour) for neighbour in neighbours]
    expression1 = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.MULTIPLY)
    assert str(expression1) in neighbour_strings
    assert str(expression2) in neighbour_strings


def test_expression_equivalence_search_space():
    search_space = CKSWithRQSearchSpace(3)
    kernel_config_1 = RBFWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=1)
    kernel_config_2 = LinearWithPriorConfig(input_dimension=3, active_on_single_dimension=True, active_dimension=2)
    base_expression_1 = ElementaryKernelGrammarExpression(KernelFactory.build(kernel_config_1))
    base_expression_2 = ElementaryKernelGrammarExpression(KernelFactory.build(kernel_config_2))
    expression1 = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(expression1, base_expression_1, KernelGrammarOperator.MULTIPLY)
    expression3 = KernelGrammarExpression(base_expression_1, expression1, KernelGrammarOperator.MULTIPLY)
    expression1_2 = KernelGrammarExpression(base_expression_1, base_expression_1, KernelGrammarOperator.MULTIPLY)
    expression2_2 = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.MULTIPLY)
    expression3_2 = KernelGrammarExpression(expression1_2, expression2_2, KernelGrammarOperator.ADD)
    assert search_space.check_expression_equality(expression2, expression3)
    assert search_space.check_expression_equality(expression3, expression3_2)
    search_space.operator_list.append(KernelGrammarOperator.SPLIT_CH)
    assert search_space.check_expression_equality(expression2, expression3)
    assert not search_space.check_expression_equality(expression3, expression3_2)


def test_invariant_hash():
    grammar_generator = GeneratorFactory.build(CompositionalKernelSearchGeneratorConfig(input_dimension=2))
    expression_list = grammar_generator.get_dataset_recursivly_generated(100, 1)
    for expression in expression_list:
        hash_value1, _ = expression.get_add_mult_invariant_hash()
        expression_normal_form = KernelGrammarExpressionTransformer.transform_to_normal_form(expression)
        hash_value2, _ = expression_normal_form.get_add_mult_invariant_hash()
        assert hash_value1 == hash_value2


def test_kernel_grammar_operator_wrapper():
    operator1 = KernelGrammarOperatorWrapper(KernelGrammarOperator.ADD)
    operator2 = KernelGrammarOperator.ADD
    operator3 = KernelGrammarOperator.MULTIPLY
    operator4 = KernelGrammarOperatorWrapper(KernelGrammarOperator.ADD)
    operator5 = KernelGrammarOperatorWrapper(KernelGrammarOperator.MULTIPLY)
    assert operator1 != operator3
    assert operator1 == operator2
    assert operator1 == operator4
    assert not operator1 == operator5


if __name__ == "__main__":
    test_kernel_grammar_operator_wrapper()
