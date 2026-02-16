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

from enum import Enum
from typing import Dict, List, Optional, Tuple
import gpflow
from gpflow.config.__config__ import default_float
from gpflow.utilities.bijectors import positive
from tensorflow_probability import distributions as tfd
import numpy as np
from alef.configs.kernels.linear_configs import LinearWithPriorConfig
from alef.configs.kernels.periodic_configs import PeriodicWithPriorConfig
from alef.configs.kernels.rational_quadratic_configs import RQWithPriorConfig
from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
from alef.kernels.base_object_kernel import BaseObjectKernel
from alef.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarExpressionTransformer,
    KernelGrammarOperator,
)
import tensorflow as tf
from alef.kernels.kernel_grammar.optimal_transport_mappings import (
    DimWiseWeightedDistanceExtractor,
    KernelGrammarTreeDistanceMapper,
)

from alef.kernels.linear_kernel import LinearKernel
from alef.kernels.rational_quadratic_kernel import RationalQuadraticKernel
from alef.kernels.rbf_kernel import RBFKernel
from alef.kernels.periodic_kernel import PeriodicKernel
from alef.utils.utils import manhatten_distance
import tensorflow_probability as tfp

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float


# Dict containing substructure information for one tree (one kernel in the kernel grammar)
# key=hash value of substructure
# List[0] - int - number of occurance of the substructure in the tree
# List[1] - object - meta info about substructure
StructuredDict = Dict[int, List]


class BaseKernelGrammarKernel(BaseObjectKernel):
    def __init__(self, base_variance: float, parameters_trainable: bool, transform_to_normal: bool, **kwargs):
        self.variance = gpflow.Parameter(f64(base_variance), transform=positive(), trainable=parameters_trainable)
        self.transform_to_normal = transform_to_normal

    def K(self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None):
        """
        Computes the kernel matrix between the given sets of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.
            X2: Optional list of kernel grammar expressions. If None, computes the kernel matrix for X with itself.

        Returns:
            Kernel matrix as a TensorFlow tensor.
        """
        if X2 is None:
            X = self.internal_transform_X(X)
            self.create_hash_array_mapping(X)
            feature_matrix = tf.convert_to_tensor(self.feature_matrix(X), dtype=default_float())
            weighting_vector = self.get_weighting_vector()
            weighted_feature_matrix = tf.math.multiply(weighting_vector, feature_matrix)
            K_values = tf.linalg.matmul(weighted_feature_matrix, tf.transpose(weighted_feature_matrix))
            diag_K = tf.expand_dims(tf.linalg.diag_part(K_values), axis=1)
            normalizer = tf.matmul(tf.sqrt(diag_K), tf.sqrt(tf.transpose(diag_K)))
            K = self.variance * (K_values / normalizer)
            return K
        else:
            X = self.internal_transform_X(X)
            X2 = self.internal_transform_X(X2)
            self.create_hash_array_mapping(X + X2)
            weighting_vector = self.get_weighting_vector()
            feature_matrix_X = tf.convert_to_tensor(self.feature_matrix(X), dtype=default_float())
            feature_matrix_X2 = tf.convert_to_tensor(self.feature_matrix(X2), dtype=default_float())
            weighted_feature_matrix_X = tf.math.multiply(weighting_vector, feature_matrix_X)
            weighted_feature_matrix_X2 = tf.math.multiply(weighting_vector, feature_matrix_X2)
            diag_K_X = tf.expand_dims(
                tf.linalg.diag_part(
                    tf.linalg.matmul(weighted_feature_matrix_X, tf.transpose(weighted_feature_matrix_X))
                ),
                axis=1,
            )
            diag_K_X2 = tf.expand_dims(
                tf.linalg.diag_part(
                    tf.linalg.matmul(weighted_feature_matrix_X2, tf.transpose(weighted_feature_matrix_X2))
                ),
                axis=1,
            )
            normalizer = tf.matmul(tf.sqrt(diag_K_X), tf.sqrt(tf.transpose(diag_K_X2)))
            K_values = tf.linalg.matmul(weighted_feature_matrix_X, tf.transpose(weighted_feature_matrix_X2))
            K = self.variance * (K_values / normalizer)
            return K

    def K_diag(self, X: List[BaseKernelGrammarExpression]):
        """
        Computes the diagonal of the kernel matrix for the given set of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            Diagonal of the kernel matrix as a TensorFlow tensor.
        """
        X = self.internal_transform_X(X)
        diag = self.variance * tf.ones(len(X), dtype=default_float())
        return diag

    def create_hash_array_mapping(self, tree_dict_list: List[StructuredDict]):
        """
        Creates a mapping from substructure hashes to feature vector indices.

        Args:
            tree_dict_list: List of dictionaries containing substructure information.
        """
        self.big_index_dict = {}
        index = 0
        for tree_dict in tree_dict_list:
            for hash in tree_dict:
                if not hash in self.big_index_dict:
                    feature_meta_info = tree_dict[hash][1]
                    self.big_index_dict[hash] = [index, feature_meta_info]
                    index += 1

    def get_feature_vector_length(self):
        """
        Returns the length of the feature vector.

        Returns:
            Length of the feature vector.
        """
        return len(self.big_index_dict)

    def feature_matrix(self, X: List[StructuredDict]) -> np.array:
        """
        Creates the feature matrix for the given set of structured dictionaries.

        Args:
            X: List of structured dictionaries.

        Returns:
            Feature matrix as a NumPy array.
        """
        raise NotImplementedError

    def get_weighting_vector(self) -> tf.Tensor:
        """
        Returns the learnable weighting vector.

        Returns:
            Weighting vector as a TensorFlow tensor.
        """
        raise NotImplementedError

    def internal_transform_X(self, X: List[BaseKernelGrammarExpression]) -> List[StructuredDict]:
        """
        Transforms the input kernel grammar expressions into structured dictionaries.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            List of structured dictionaries.
        """
        raise NotImplementedError

    def transform_X(self, X: List[BaseKernelGrammarExpression]) -> List[BaseKernelGrammarExpression]:
        """
        Transforms the input kernel grammar expressions to their normal form if required.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            List of transformed kernel grammar expressions.
        """
        if self.transform_to_normal:
            new_X = []
            for x in X:
                x_normal_form = KernelGrammarExpressionTransformer.transform_to_normal_form(x)
                new_X.append(x_normal_form)
            return new_X
        return X


class SumKernelKernelGrammarTree(BaseKernelGrammarKernel):
    def __init__(self, kernel_kernel_list: List[BaseKernelGrammarKernel], transform_to_normal: bool, **kwargs):
        super().__init__(1.0, False, transform_to_normal)
        self.kernel_kernel_list = kernel_kernel_list

    def K(
        self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None
    ) -> tf.Tensor:
        """
        Computes the sum of kernel matrices from the list of kernel kernels.

        Args:
            X: List of kernel grammar expressions.
            X2: Optional list of kernel grammar expressions. If None, computes the kernel matrix for X with itself.

        Returns:
            Summed kernel matrix as a TensorFlow tensor.
        """
        if X2 is None:
            return tf.add_n([k.K(X, X2=None) for k in self.kernel_kernel_list])
        else:
            return tf.add_n([k.K(X, X2) for k in self.kernel_kernel_list])

    def K_diag(self, X: List[BaseKernelGrammarExpression]) -> tf.Tensor:
        """
        Computes the diagonal of the summed kernel matrix for the given set of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            Diagonal of the summed kernel matrix as a TensorFlow tensor.
        """
        return tf.add_n([k.K_diag(X) for k in self.kernel_kernel_list])


class MultiplyKernelGrammarKernels(BaseKernelGrammarKernel):
    def __init__(
        self, kernel_1: BaseKernelGrammarKernel, kernel_2: BaseKernelGrammarKernel, transform_to_normal: bool, **kwargs
    ):
        super().__init__(1.0, False, transform_to_normal)
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def K(
        self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None
    ) -> tf.Tensor:
        """
        Computes the product of kernel matrices from the two kernel kernels.

        Args:
            X: List of kernel grammar expressions.
            X2: Optional list of kernel grammar expressions. If None, computes the kernel matrix for X with itself.

        Returns:
            Product kernel matrix as a TensorFlow tensor.
        """
        if X2 is None:
            return tf.multiply(self.kernel_1.K(X), self.kernel_2.K(X))
        else:
            return tf.multiply(self.kernel_1.K(X, X2), self.kernel_2.K(X, X2))

    def K_diag(self, X: List[BaseKernelGrammarExpression]) -> tf.Tensor:
        """
        Computes the diagonal of the product kernel matrix for the given set of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            Diagonal of the product kernel matrix as a TensorFlow tensor.
        """
        return tf.multiply(self.kernel_1.K_diag(X), self.kernel_2.K_diag(X))


class KernelGrammarSubtreeKernel(BaseKernelGrammarKernel):
    def __init__(self, base_variance: float, parameters_trainable: bool, transform_to_normal: bool, **kwargs):
        super().__init__(base_variance, parameters_trainable, transform_to_normal)
        # self.lamb = gpflow.Parameter(f64(0.5), transform=positive(), trainable=parameters_trainable)
        transform = tfp.bijectors.Sigmoid(low=None, high=None, validate_args=False, name="sigmoid")
        self.lamb = gpflow.Parameter(f64(0.5), transform=transform, trainable=parameters_trainable)

    def get_weighting_vector(self) -> tf.Tensor:
        """
        Returns the learnable weighting vector based on the subtree sizes.

        Returns:
            Weighting vector as a TensorFlow tensor.
        """
        p_vec = tf.convert_to_tensor(self.get_sub_tree_size_vector(), dtype=default_float())
        weighting_vector = tf.sqrt(tf.math.pow(self.lamb, p_vec))
        return weighting_vector

    def feature_matrix(self, X: List[StructuredDict]) -> np.array:
        """
        Creates the feature matrix for the given set of structured dictionaries.

        Args:
            X: List of structured dictionaries.

        Returns:
            Feature matrix as a NumPy array.
        """
        feature_matrix = []
        for subtree_dict in X:
            feature_vec = self.subtree_dict_to_feature_vector(subtree_dict)
            feature_matrix.append(feature_vec)
        return np.array(feature_matrix)

    def get_sub_tree_size_vector(self):
        """
        Returns the vector of subtree sizes.

        Returns:
            Vector of subtree sizes as a NumPy array.
        """
        p_vec = np.zeros(len(self.big_index_dict))
        for key in self.big_index_dict:
            subtree_meta_info = self.big_index_dict[key][1]
            p_vec[self.big_index_dict[key][0]] = subtree_meta_info.num_elementary
        return p_vec

    def subtree_dict_to_feature_vector(self, subtree_dict_kernel: StructuredDict):
        """
        Converts a subtree dictionary to a feature vector.

        Args:
            subtree_dict_kernel: Dictionary containing subtree information.

        Returns:
            Feature vector as a NumPy array.
        """
        feature_vec = np.zeros(len(self.big_index_dict))
        for key in subtree_dict_kernel:
            feature_vec[self.big_index_dict[key][0]] = subtree_dict_kernel[key][0]
        return feature_vec

    def internal_transform_X(self, X: List[BaseKernelGrammarExpression]) -> List[StructuredDict]:
        """
        Transforms the input kernel grammar expressions into structured dictionaries.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            List of structured dictionaries.
        """
        dict_list = []
        for kernel_grammar_expression in X:
            subtree_dict = kernel_grammar_expression.get_subtree_dict()
            dict_list.append(subtree_dict)
        return dict_list


class FeatureType(Enum):
    ELEMENTARY_COUNT = 0
    CONTAINS_ELEMENTARY = 1
    ELEMENTARIES_UNDER_ADD = 2
    ELEMENTARIES_UNDER_MULT = 3
    SUBTREES = 4
    ONE_GRAM_TREE_METRIC = 5
    REDUCED_ELEMENTARY_PATHS = 6
    DIM_WISE_WEIGHTED_ELEMENTARY_COUNT = 7
    ADD_MULT_INVARIANT_SUBTREES = 8
    SUBTREES_WITHOUT_LEAFS = 9


class OptimalTransportKernelKernel(BaseKernelGrammarKernel):
    """
    Main kernel kernel implementation for the project "Structural Kernel Search via BO and SOT". It extracts features (specified in FeatureType)
    from kernel grammar expressions and calculates pair-wise manhatten-distances between those features. Multiple features can be extracted - final distance is
    than weighted over features distances.

    Attributes:
        feature_type_list: list of FeatureType enums specifying which features should be extracted from the BaseKernelGrammarExpressions
        base_variance: (initial) value of variance of the kernel-kernel - usually gets fitted
        base_lengthscale:  (initial) value of lengthscale of the kernel-kernel - usually gets fitted
        base_alpha:  (initial) value of alpha of the kernel-kernel - usually gets fitted
        alpha_trainable: flag if alphas are trainable
        parameters_trainable: flag if kernel parameters in general are trainable - alphas only are trainable if both flags are true
        transform_to_normal: flag if grammar expression should be transformed to a normal form (if possible) before evaluating the kernel-kernel on it
        use_hyperprior: flag if hyperprior should be used for variance and lengthscale
        lengthscale_prior_parameters: tuple of hyperprior gamma dist of lengthscale
        variance_prior_parameters: tuple of hyperprior gamma dist of variance
    """

    def __init__(
        self,
        feature_type_list: List[FeatureType],
        base_variance: float,
        base_lengthscale: float,
        base_alpha: float,
        alpha_trainable: bool,
        parameters_trainable: bool,
        transform_to_normal: bool,
        use_hyperprior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        **kwargs,
    ):
        super().__init__(1.0, False, transform_to_normal)
        transform = tfp.bijectors.Sigmoid(low=None, high=None, validate_args=False, name="sigmoid")
        train_alpha = parameters_trainable and alpha_trainable
        self.alphas = gpflow.Parameter(
            f64(np.repeat(base_alpha, len(feature_type_list))), transform=transform, trainable=train_alpha
        )
        self.lengthscale = gpflow.Parameter(f64(base_lengthscale), transform=positive(), trainable=parameters_trainable)
        self.variance = gpflow.Parameter(f64(base_variance), transform=positive(), trainable=parameters_trainable)
        if use_hyperprior:
            ls_prior_mu, ls_prior_sigma = lengthscale_prior_parameters
            v_prior_mu, v_prior_sigma = variance_prior_parameters
            self.lengthscale.prior = tfd.LogNormal(loc=f64(ls_prior_mu), scale=f64(ls_prior_sigma))
            self.variance.prior = tfd.LogNormal(loc=f64(v_prior_mu), scale=f64(v_prior_sigma))
        self.feature_type_list = feature_type_list

    def K(self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None):
        """
        Computes the kernel matrix using the Wasserstein distance between the given sets of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.
            X2: Optional list of kernel grammar expressions. If None, computes the kernel matrix for X with itself.

        Returns:
            Kernel matrix as a TensorFlow tensor.
        """
        distance_matrix = self.wasserstein_distance(X, X2)
        K = self.variance * tf.math.exp(-1 * distance_matrix / tf.pow(self.lengthscale, 2.0))
        return K

    def K_diag(self, X: List[BaseKernelGrammarExpression]):
        """
        Computes the diagonal of the kernel matrix using the Wasserstein distance for the given set of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            Diagonal of the kernel matrix as a TensorFlow tensor.
        """
        distance_matrix = self.wasserstein_distance(X)
        diag_distance = tf.linalg.diag_part(distance_matrix)
        K_diag = self.variance * tf.math.exp(-1 * diag_distance / tf.pow(self.lengthscale, 2.0))
        return K_diag

    def get_distance_matrix(
        self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None
    ):
        """
        Computes the distance matrix using the Wasserstein distance between the given sets of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.
            X2: Optional list of kernel grammar expressions. If None, computes the distance matrix for X with itself.

        Returns:
            Distance matrix as a TensorFlow tensor.
        """
        return self.wasserstein_distance(X, X2)

    def get_manhatten_distances(
        self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None
    ):
        """
        Computes the Manhattan distances between the given sets of kernel grammar expressions for each feature type.

        Args:
            X: List of kernel grammar expressions.
            X2: Optional list of kernel grammar expressions. If None, computes the distances for X with itself.

        Returns:
            List of Manhattan distance matrices for each feature type.
        """
        if X2 is None:
            manhatten_distances = []
            for feature_type in self.feature_type_list:
                X_feature = self.internal_transform_X(X, feature_type)
                index_dict_feature = self.create_hash_array_mapping(X_feature)
                feature_matrix = tf.convert_to_tensor(
                    self.feature_matrix(X_feature, index_dict_feature, normalize=self.normalize_features(feature_type)),
                    dtype=default_float(),
                )
                manhatten_distance_feature = manhatten_distance(feature_matrix, feature_matrix)
                manhatten_distances.append(manhatten_distance_feature)
        else:
            manhatten_distances = []
            for feature_type in self.feature_type_list:
                X_feature = self.internal_transform_X(X, feature_type)
                X2_feature = self.internal_transform_X(X2, feature_type)
                index_dict_feature = self.create_hash_array_mapping(X_feature + X2_feature)
                X_feature_matrix = tf.convert_to_tensor(
                    self.feature_matrix(X_feature, index_dict_feature, normalize=self.normalize_features(feature_type)),
                    dtype=default_float(),
                )
                X2_feature_matrix = tf.convert_to_tensor(
                    self.feature_matrix(
                        X2_feature, index_dict_feature, normalize=self.normalize_features(feature_type)
                    ),
                    dtype=default_float(),
                )
                manhatten_distance_feature = manhatten_distance(X_feature_matrix, X2_feature_matrix)
                manhatten_distances.append(manhatten_distance_feature)
        return manhatten_distances

    def wasserstein_distance(
        self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None
    ):
        """
        Computes the Wasserstein distance between the given sets of kernel grammar expressions.

        Args:
            X: List of kernel grammar expressions.
            X2: Optional list of kernel grammar expressions. If None, computes the distance for X with itself.

        Returns:
            Wasserstein distance matrix as a TensorFlow tensor.
        """
        manhatten_distances = self.get_manhatten_distances(X, X2)
        distance_weighting = tf.expand_dims(tf.expand_dims(self.alphas / tf.reduce_sum(self.alphas), axis=1), axis=2)
        manhatten_distances_stacked = tf.stack(manhatten_distances)
        distance_matrix = tf.reduce_sum(tf.multiply(distance_weighting, manhatten_distances_stacked), axis=0)
        return distance_matrix

    def internal_transform_X(
        self, X: List[BaseKernelGrammarExpression], feature_type: FeatureType
    ) -> List[StructuredDict]:
        """
        Extracts the feature vectors from the list of kernel grammar expressions based on the specified feature type.

        Args:
            X: List of kernel grammar expressions.
            feature_type: Feature type to extract.

        Returns:
            List of structured dictionaries containing the extracted features.
        """
        dict_list = []
        if feature_type == FeatureType.ONE_GRAM_TREE_METRIC:
            tree_distance_mapper = KernelGrammarTreeDistanceMapper(
                X[0].get_generator_name(), X[0].get_input_dimension()
            )
        if feature_type == FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT:
            dim_wise_mapper = DimWiseWeightedDistanceExtractor(X[0].get_generator_name(), X[0].get_input_dimension())

        for kernel_grammar_expression in X:
            if feature_type == FeatureType.ELEMENTARY_COUNT:
                feature_dict = kernel_grammar_expression.get_elementary_count_dict()
            elif feature_type == FeatureType.CONTAINS_ELEMENTARY:
                feature_dict = kernel_grammar_expression.get_contains_elementary_dict()
            elif feature_type == FeatureType.ELEMENTARIES_UNDER_ADD:
                feature_dict = kernel_grammar_expression.get_elementary_below_operator_dict(KernelGrammarOperator.ADD)
            elif feature_type == FeatureType.ELEMENTARIES_UNDER_MULT:
                feature_dict = kernel_grammar_expression.get_elementary_below_operator_dict(
                    KernelGrammarOperator.MULTIPLY
                )
            elif feature_type == FeatureType.SUBTREES:
                feature_dict = kernel_grammar_expression.get_subtree_dict()
            elif feature_type == FeatureType.REDUCED_ELEMENTARY_PATHS:
                feature_dict = kernel_grammar_expression.get_elementary_path_dict(
                    [KernelGrammarOperator.ADD, KernelGrammarOperator.MULTIPLY, KernelGrammarOperator.SPLIT_CH]
                )
            elif feature_type == FeatureType.ONE_GRAM_TREE_METRIC:
                feature_dict = tree_distance_mapper.get_weighted_feature_dict(kernel_grammar_expression)
            elif feature_type == FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT:
                feature_dict = dim_wise_mapper.get_dim_wise_weighted_elementary_features(kernel_grammar_expression)
            elif feature_type == FeatureType.ADD_MULT_INVARIANT_SUBTREES:
                feature_dict = kernel_grammar_expression.get_add_mult_invariant_subtree_dict()
            elif feature_type == FeatureType.SUBTREES_WITHOUT_LEAFS:
                feature_dict = kernel_grammar_expression.get_subtree_dict_without_leafs()
            self.check_feature_dict(feature_dict)
            dict_list.append(feature_dict)
        return dict_list

    def check_feature_dict(self, feature_dict):
        """
        Checks the validity of the feature dictionary.

        Args:
            feature_dict: Dictionary containing feature information.

        Raises:
            AssertionError: If the feature dictionary is invalid.
        """
        has_value = False
        for key in feature_dict:
            if isinstance(feature_dict[key], list):
                if feature_dict[key][0] > 0:
                    has_value = True
            else:
                if feature_dict[key] > 0:
                    has_value = True
        assert has_value

    def normalize_features(self, feature_type):
        """
        Determines whether the features should be normalized based on the feature type.

        Args:
            feature_type: Feature type to check.

        Returns:
            Boolean indicating whether the features should be normalized.
        """
        if feature_type == FeatureType.ONE_GRAM_TREE_METRIC:
            return False
        elif feature_type == FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT:
            return False
        return True

    def create_hash_array_mapping(self, tree_dict_list: List[StructuredDict]) -> StructuredDict:
        """
        Creates a mapping from substructure hashes to feature vector indices.

        Args:
            tree_dict_list: List of dictionaries containing substructure information.

        Returns:
            Dictionary mapping substructure hashes to feature vector indices.
        """
        index_dict = {}
        index = 0
        for tree_dict in tree_dict_list:
            for hash in tree_dict:
                if not hash in index_dict:
                    index_dict[hash] = [index]
                    index += 1
        return index_dict

    def feature_matrix(self, X: List[StructuredDict], index_dict: StructuredDict, normalize: bool) -> np.array:
        """
        Transforms feature dictionaries to a numerical feature matrix.

        Args:
            X: List of structured dictionaries.
            index_dict: Dictionary mapping substructure hashes to feature vector indices.
            normalize: Boolean indicating whether to normalize the features.

        Returns:
            Feature matrix as a NumPy array.
        """
        feature_matrix = []
        for feature_dict in X:
            feature_vec = self.feature_dict_to_feature_vector(feature_dict, index_dict)
            if normalize:
                feature_vec = feature_vec / np.sum(feature_vec)
            feature_matrix.append(feature_vec)
        return np.array(feature_matrix)

    def feature_dict_to_feature_vector(self, feature_dict: StructuredDict, index_dict: StructuredDict):
        """
        Converts a feature dictionary to a feature vector.

        Args:
            feature_dict: Dictionary containing feature information.
            index_dict: Dictionary mapping substructure hashes to feature vector indices.

        Returns:
            Feature vector as a NumPy array.
        """
        feature_vec = np.zeros(len(index_dict))
        for key in feature_dict:
            if isinstance(feature_dict[key], list):
                assert isinstance(feature_dict[key][0], int)
                feature_vec[index_dict[key][0]] = feature_dict[key][0]
            else:
                feature_vec[index_dict[key][0]] = feature_dict[key]
        return feature_vec

    def transform_X(self, X: List[BaseKernelGrammarExpression]) -> List[BaseKernelGrammarExpression]:
        """
        Transforms the input kernel grammar expressions to their normal form if required.

        Args:
            X: List of kernel grammar expressions.

        Returns:
            List of transformed kernel grammar expressions.
        """
        return super().transform_X(X)


if __name__ == "__main__":
    # kernel_kernel = StaticFeaturesKernel(1.0, FeatureType.SUBTREES, True)
    kernel_kernel = OptimalTransportKernelKernel(
        [FeatureType.ELEMENTARY_COUNT, FeatureType.SUBTREES],
        1.0,
        1.0,
        0.5,
        False,
        False,
        False,
        False,
        (1.0, 1.0),
        (1.0, 1.0),
    )

    base_expression_1 = ElementaryKernelGrammarExpression(
        RBFKernel(**RBFWithPriorConfig(input_dimension=2, active_on_single_dimension=True).dict())
    )
    base_expression_2 = ElementaryKernelGrammarExpression(
        LinearKernel(**LinearWithPriorConfig(input_dimension=2, active_on_single_dimension=True).dict())
    )
    base_expression_3 = ElementaryKernelGrammarExpression(
        RationalQuadraticKernel(**RQWithPriorConfig(input_dimension=2, active_on_single_dimension=True).dict())
    )
    base_expression_4 = ElementaryKernelGrammarExpression(
        RBFKernel(**RBFWithPriorConfig(input_dimension=2, active_on_single_dimension=True).dict())
    )
    base_expression_5 = ElementaryKernelGrammarExpression(
        PeriodicKernel(**PeriodicWithPriorConfig(input_dimension=2, active_on_single_dimension=True).dict())
    )

    if True:
        expression = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
        expression2 = KernelGrammarExpression(expression, base_expression_3, KernelGrammarOperator.ADD)
        expression3 = KernelGrammarExpression(base_expression_4, expression2, KernelGrammarOperator.MULTIPLY)
        expression4 = KernelGrammarExpression(expression2, expression3, KernelGrammarOperator.MULTIPLY)

    print(expression3)
    print(expression4)
    print("")
    print(expression3.get_elementary_path_dict([KernelGrammarOperator.ADD]))
    print("")
    print(expression4.get_elementary_path_dict([KernelGrammarOperator.ADD]))

    X = [expression3, expression4, base_expression_3, expression2, base_expression_5, base_expression_2]
    X2 = [expression2, base_expression_4]
    for x in X:
        assert isinstance(x, BaseKernelGrammarExpression)
        x.set_generator_name("CKSWithRQGenerator")
