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

import tensorflow as tf
import numpy as np
from typing import List, Optional
import gpflow
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfd
from alef.models.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from alef.utils.utils import k_means

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float


class ResidualLayer(tf.Module):
    @staticmethod
    def create_single_weight_and_bias(
        W_n_rows: int, W_n_cols: int, b_size: int, W_scale: float, b_scale: float, add_priors: bool = False
    ):
        W = gpflow.Parameter(tf.random.normal([W_n_rows, W_n_cols], stddev=W_scale), trainable=True)
        b = gpflow.Parameter(tf.random.normal([b_size], stddev=b_scale), trainable=True)
        if add_priors:
            W.prior = tfd.Normal(np.zeros([W_n_rows, W_n_cols]), W_scale * np.ones([W_n_rows, W_n_cols]))
            b.prior = tfd.Normal(np.zeros([b_size]), b_scale * np.ones([b_size]))
        print(W.shape)
        return W, b

    @staticmethod
    def create_weight_and_bias_list_from_layer_list(
        dimension: int, layer_size_list: List[int], W_scale: float, b_scale: float, add_prior: bool = False
    ):
        weight_list = []
        bias_list = []
        W, b = ResidualLayer.create_single_weight_and_bias(
            dimension, layer_size_list[0], layer_size_list[0], W_scale, b_scale, add_prior
        )
        weight_list.append(W)
        bias_list.append(b)
        for i in range(1, len(layer_size_list)):
            W, b = ResidualLayer.create_single_weight_and_bias(
                layer_size_list[i - 1], layer_size_list[i], layer_size_list[i], W_scale, b_scale, add_prior
            )
            weight_list.append(W)
            bias_list.append(b)
        W, b = ResidualLayer.create_single_weight_and_bias(
            layer_size_list[-1], dimension, dimension, W_scale, b_scale, add_prior
        )
        weight_list.append(W)
        bias_list.append(b)
        return weight_list, bias_list

    def __init__(
        self,
        dimension: int,
        set_weights: bool,
        layer_size_list: List[int],
        weight_list: Optional[List[tf.Variable]],
        bias_list: Optional[List[tf.Variable]],
        W_scale: float,
        b_scale: float,
        add_prior: bool = False,
    ):
        self.dimension = dimension
        self.c = 0.95
        if set_weights:
            self.weight_list = weight_list
            self.bias_list = bias_list
        else:
            self.weight_list, self.bias_list = ResidualLayer.create_weight_and_bias_list_from_layer_list(
                self.dimension, layer_size_list, W_scale, b_scale, add_prior
            )

    def forward(self, X):
        out = self.mlp(X)
        return out + X

    def mlp(self, X):
        out = self.mlp_layer(X, self.weight_list[0], self.bias_list[0])
        for i in range(1, len(self.weight_list) - 1):
            out = self.mlp_layer(out, self.weight_list[i], self.bias_list[i])
        out = self.linear_layer(out, self.weight_list[-1], self.bias_list[-1])
        return out

    def linear_layer(self, X, W, b):
        W_norm = tf.norm(W, ord="euclidean")
        alpha = tf.clip_by_value(self.c / W_norm, clip_value_min=0.0, clip_value_max=1.0)
        out = tf.matmul(X, alpha * W) + b
        return out

    def mlp_layer(self, X, W, b):
        W_norm = tf.norm(W, ord="euclidean")
        alpha = tf.clip_by_value(self.c / W_norm, clip_value_min=0.0, clip_value_max=1.0)
        out = tf.nn.relu(tf.matmul(X, alpha * W) + b)
        return out


class InvertibleResNet(BaseFeatureExtractor, tf.Module):
    def __init__(
        self,
        input_dimension: int,
        num_layers: int,
        residual_layer_size_list: List[int],
        share_weights: bool,
        W_scale: float,
        b_scale: float,
        add_prior: bool,
        add_regularization: bool,
        regularizer_lambdas: List[float],
        contrain_regularizer_n_data: bool,
        contrained_regularizer_k: int,
        add_noise_on_train_mode: bool,
        layer_noise_std: float,
        **kwargs,
    ):
        self.dimension = input_dimension
        self.num_layers = num_layers
        self.residual_layer_list = []
        weight_list = []
        bias_list = []
        weight_list = []
        bias_list = []
        self.add_regularization = add_regularization
        self.regularizer_lambdas = regularizer_lambdas
        self.contrain_regularizer_n_data = contrain_regularizer_n_data
        self.contrained_regularizer_k = contrained_regularizer_k
        self.training_mode = False
        self.add_noise_on_train_mode = add_noise_on_train_mode
        self.layer_noise_std = layer_noise_std
        if share_weights:
            weight_list, bias_list = ResidualLayer.create_weight_and_bias_list_from_layer_list(
                self.dimension, residual_layer_size_list, W_scale, b_scale, add_prior
            )
        for i in range(0, num_layers):
            residual_layer = ResidualLayer(
                self.dimension,
                set_weights=share_weights,
                layer_size_list=residual_layer_size_list,
                weight_list=weight_list,
                bias_list=bias_list,
                W_scale=W_scale,
                b_scale=b_scale,
                add_prior=add_prior,
            )
            self.residual_layer_list.append(residual_layer)

    def get_input_dimension(self) -> int:
        return self.dimension

    def get_output_dimension(self) -> int:
        return self.dimension

    def forward(self, X) -> tf.Tensor:
        out = self.residual_layer_list[0].forward(X)
        for i in range(1, len(self.residual_layer_list)):
            out = self.residual_layer_list[i].forward(out)
        if self.training_mode and self.add_noise_on_train_mode:
            out = out + tf.random.normal(out.shape, stddev=self.layer_noise_std, dtype=gpflow.default_float())
        return out

    def regularization_loss(self, x_data: np.array) -> tf.Tensor:
        if self.add_regularization:
            (
                diff_norm,
                jacob_norm,
                divergence_sum,
                off_diag_jacob_norm,
                rotation_elements_norm,
                distance_difference_capped_sum,
            ) = self.regularizer_losses(x_data, self.contrain_regularizer_n_data, self.contrained_regularizer_k)
            loss = (
                self.regularizer_lambdas[0] * diff_norm
                + self.regularizer_lambdas[1] * jacob_norm
                + self.regularizer_lambdas[2] * divergence_sum
                + self.regularizer_lambdas[3] * off_diag_jacob_norm
                + self.regularizer_lambdas[4] * rotation_elements_norm
                + self.regularizer_lambdas[5] * distance_difference_capped_sum
            )
        else:
            loss = np.double(0.0)
        return loss

    def regularizer_losses(self, X, constrain_n_data_to_k=False, k=50, l_distance=4.0):
        if constrain_n_data_to_k:
            if X.shape[0] > k:
                X = k_means(k, X)
        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X)
        len_X = X.shape[0]
        dim_x = X.shape[1]
        out = X
        for i in range(0, len(self.residual_layer_list)):
            with tf.GradientTape() as tape:
                input = out
                tape.watch(input)
                diff = self.residual_layer_list[i].mlp(input)
            diff_jacobian = tape.batch_jacobian(diff, input)
            divergences = tf.reduce_sum(tf.linalg.diag_part(diff_jacobian), axis=1)
            off_diagonals_jacob = tf.linalg.set_diag(diff_jacobian, np.zeros((len_X, dim_x)))
            off_diagonals_jacob_transposed = tf.transpose(off_diagonals_jacob, perm=[0, 2, 1])
            rotation_elements = off_diagonals_jacob - off_diagonals_jacob_transposed
            out = diff + input
            if i == 0:
                diff_norm = tf.reduce_sum(tf.pow(tf.norm(diff, axis=1), 2.0))
                jacob_norm = tf.pow(tf.norm(diff_jacobian), 2.0)
                off_diag_jacob_norm = tf.pow(tf.norm(off_diagonals_jacob), 2.0)
                divergence_sum = tf.reduce_sum(divergences)
                rotation_elements_norm = tf.pow(tf.norm(rotation_elements), 2.0)
            else:
                diff_norm += tf.reduce_sum(tf.pow(tf.norm(diff, axis=1), 2.0))
                jacob_norm += tf.pow(tf.norm(diff_jacobian), 2.0)
                divergence_sum += tf.reduce_sum(divergences)
                off_diag_jacob_norm += tf.pow(tf.norm(off_diagonals_jacob), 2.0)
                rotation_elements_norm += tf.pow(tf.norm(rotation_elements), 2.0)
        for d in range(0, dim_x):
            distance_difference_1 = gpflow.utilities.ops.square_distance(
                X[:, d], X[:, d]
            ) - l_distance * gpflow.utilities.ops.square_distance(out[:, d], out[:, d])
            distance_difference_2 = gpflow.utilities.ops.square_distance(
                out[:, d], out[:, d]
            ) - l_distance * gpflow.utilities.ops.square_distance(X[:, d], X[:, d])
            if d == 0:
                distance_difference_capped_sum = tf.reduce_sum(
                    tf.math.maximum(distance_difference_1, 0.0)
                ) + tf.reduce_sum(tf.math.maximum(distance_difference_2, 0.0))
            else:
                distance_difference_capped_sum += tf.reduce_sum(
                    tf.math.maximum(distance_difference_1, 0.0)
                ) + tf.reduce_sum(tf.math.maximum(distance_difference_2, 0.0))

        return (
            diff_norm,
            jacob_norm,
            divergence_sum,
            off_diag_jacob_norm,
            rotation_elements_norm,
            distance_difference_capped_sum,
        )

    def plot_warping(self, X):
        input_dimension = X.shape[1]
        if input_dimension == 1:
            out = self.forward(X)
            trajectory_array = np.transpose(np.array([np.squeeze(X), np.squeeze(out)]))
            time_array = np.array([0.0, 1.0])
            fig, ax = plt.subplots()
            for trajectory in trajectory_array:
                ax.plot(time_array, trajectory)
            plt.show()
        elif input_dimension == 2:
            out = self.forward(X)
            fig, ax = plt.subplots()
            ax.scatter(out[:, 0], out[:, 1])
            plt.show()

    def set_mode(self, training_mode: bool):
        self.training_mode = training_mode
