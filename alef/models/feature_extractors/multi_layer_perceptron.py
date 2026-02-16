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

from alef.models.feature_extractors.base_feature_extractor import BaseFeatureExtractor
import gpflow
import numpy as np
from tensorflow_probability import distributions as tfd
import tensorflow as tf
from typing import List


class DenseLayer(tf.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        add_prior: bool,
        prior_w_sigma: float,
        prior_b_sigma: float,
        is_linear_layer: bool = False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = gpflow.Parameter(tf.random.normal([self.input_dim, self.output_dim]), trainable=True)
        self.b = gpflow.Parameter(tf.zeros([self.output_dim]), trainable=True)
        self.is_linear_layer = is_linear_layer
        if add_prior:
            self.W.prior = tfd.Normal(
                np.zeros([self.input_dim, self.output_dim]), prior_w_sigma * np.ones([self.input_dim, self.output_dim])
            )
            self.b.prior = tfd.Normal(np.zeros([self.output_dim]), prior_b_sigma * np.ones([self.output_dim]))

    def set_to_linear_layer(self):
        self.is_linear_layer = True

    def forward(self, X):
        if self.is_linear_layer:
            out = tf.matmul(X, self.W) + self.b
        else:
            out = tf.nn.relu(tf.matmul(X, self.W) + self.b)
        return out


class MultiLayerPerceptron(BaseFeatureExtractor, tf.Module):
    def __init__(
        self,
        input_dimension: int,
        layer_size_list: List[int],
        prior_w_sigma: float,
        prior_b_sigma: float,
        add_prior: bool,
        **kwargs,
    ) -> None:
        assert len(layer_size_list) >= 1
        self.input_dimension = input_dimension

        self.output_dimension = layer_size_list[-1]
        self.prior_w_sigma = prior_w_sigma
        self.prior_b_sigma = prior_b_sigma
        first_layer = DenseLayer(
            self.input_dimension,
            layer_size_list[0],
            add_prior=add_prior,
            prior_w_sigma=self.prior_w_sigma,
            prior_b_sigma=self.prior_b_sigma,
        )
        self.layer_list = [first_layer]
        for i in range(1, len(layer_size_list)):
            next_layer = DenseLayer(
                layer_size_list[i - 1],
                layer_size_list[i],
                add_prior=add_prior,
                prior_w_sigma=self.prior_w_sigma,
                prior_b_sigma=self.prior_b_sigma,
            )
            self.layer_list.append(next_layer)
        self.layer_list[-1].set_to_linear_layer()

    def forward(self, X: np.array) -> tf.Tensor:
        out = X
        for layer in self.layer_list:
            out = layer.forward(out)
        return out

    def get_input_dimension(self) -> int:
        return self.input_dimension

    def get_output_dimension(self) -> int:
        return self.output_dimension

    def regularization_loss(self, x_data: np.array) -> tf.Tensor:
        return np.double(0.0)

    def set_mode(self, training_mode: bool):
        pass
