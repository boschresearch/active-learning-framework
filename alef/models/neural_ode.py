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
import tensorflow_probability as tfp
import numpy as np
from typing import List
import gpflow


class NeuralODE(tf.Module):
    def __init__(
        self,
        layer_size_list: List[int],
        state_space_dimension: int,
        include_time_in_dynamics: bool,
        use_gpflow_parameters=False,
    ):
        assert len(layer_size_list) >= 1
        self.layer_size_list = layer_size_list
        self.include_time_in_dynamics = include_time_in_dynamics
        self.state_space_dimension = state_space_dimension
        self.polynomial_order = 10
        self.key_list = [("W" + str(i), "b" + str(i)) for i in range(0, len(self.layer_size_list) + 1)]
        self.variable_dict = {}
        if use_gpflow_parameters:
            variable_class = gpflow.Parameter
        else:
            variable_class = tf.Variable
        if self.include_time_in_dynamics:
            self.variable_dict[self.key_list[0][0]] = variable_class(
                tf.random.normal([self.polynomial_order * (self.state_space_dimension + 1), self.layer_size_list[0]])
            )
        else:
            self.variable_dict[self.key_list[0][0]] = variable_class(
                tf.random.normal([self.state_space_dimension, self.layer_size_list[0]])
            )
        self.variable_dict[self.key_list[0][1]] = variable_class(tf.random.normal([self.layer_size_list[0]]))
        for i in range(1, len(self.layer_size_list)):
            self.variable_dict[self.key_list[i][0]] = variable_class(
                tf.random.normal([self.layer_size_list[i - 1], self.layer_size_list[i]])
            )
            self.variable_dict[self.key_list[i][1]] = variable_class(tf.random.normal([self.layer_size_list[i]]))
        self.variable_dict[self.key_list[-1][0]] = variable_class(
            tf.random.normal([self.layer_size_list[-1], self.state_space_dimension])
        )
        self.variable_dict[self.key_list[-1][1]] = variable_class(tf.random.normal([self.state_space_dimension]))
        print(self.variable_dict)

    def layer(self, X, W, b, is_linear=False):
        if is_linear:
            out = tf.matmul(X, W) + b
        else:
            out = tf.nn.sigmoid(tf.matmul(X, W) + b)
        return out

    def ode(self, t, y, **constants):
        W = constants[self.key_list[0][0]]
        b = constants[self.key_list[0][1]]
        if self.include_time_in_dynamics:
            len_y = y.shape[0]
            ts = tf.expand_dims(tf.repeat(t, len_y), axis=1)
            input_list = [y, ts]
            for p in range(2, self.polynomial_order + 1):
                input_list.append(tf.pow(y, float(p)))
                input_list.append(tf.pow(ts, float(p)))
            net_input = tf.concat(input_list, axis=1)
        else:
            net_input = y
        out = self.layer(net_input, W, b)
        for i in range(1, len(self.layer_size_list)):
            W = constants[self.key_list[i][0]]
            b = constants[self.key_list[i][1]]
            out = self.layer(out, W, b)
        W = constants[self.key_list[-1][0]]
        b = constants[self.key_list[-1][1]]
        out = self.layer(out, W, b, is_linear=True)
        return out

    def forward(self, solution_times: List[float], y_0: np.array):
        y_0 = tf.constant(y_0, dtype=tf.float32)
        results = tfp.math.ode.BDF().solve(
            self.ode, solution_times[0], y_0, solution_times=solution_times, constants=self.variable_dict
        )
        return results.states

    def train(self, y0, y1, t0, t1, iterations, learning_rate):
        y1 = tf.constant(y1, dtype=tf.float32)
        for step in range(0, iterations):
            with tf.GradientTape() as tape:
                current_loss = tf.reduce_mean(tf.square(y1 - self.forward([t0, t1], y0)[-1]))
            print("Step " + str(step) + " Loss: " + str(current_loss))
            gradients = tape.gradient(current_loss, self.trainable_variables)
            for i, variable in enumerate(self.trainable_variables):
                variable.assign_sub(learning_rate * gradients[i])
