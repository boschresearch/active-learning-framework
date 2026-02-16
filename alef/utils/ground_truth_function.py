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
import numpy as np
import matplotlib.pyplot as plt


class FunctionType(Enum):
    SIN_LIN = 1
    LIN_SIN = 2
    LIN_SIN_QUAD = 3
    SAFETY_FUNCTION = 4
    SAFETY_FUNCTION_HARD = 5


class GroundTruthFunction:
    def __init__(self, function_type, noise_level):
        self.function_type = function_type
        self.noise_level = noise_level

    def f(self, x):
        if self.function_type == FunctionType.SIN_LIN:
            return self.sin_lin(x)
        elif self.function_type == FunctionType.LIN_SIN:
            return self.lin_sin(x)
        elif self.function_type == FunctionType.LIN_SIN_QUAD:
            return self.lin_sin_quad(x)
        elif self.function_type == FunctionType.SAFETY_FUNCTION:
            return self.safety_function(x)
        elif self.function_type == FunctionType.SAFETY_FUNCTION_HARD:
            return self.safety_function_hard(x)

    def f_noisy_batch(self, x):
        n = x.shape[0]
        epsilon = np.random.normal(0, self.noise_level, n)
        return self.f(x) + epsilon

    def f_noisy(self, x):
        epsilon = np.random.normal(0, self.noise_level, 1)
        return self.f(x) + epsilon

    def sin_lin(self, x):
        return x * np.sin(x)

    def lin_sin(self, x):
        return 0.5 * x + 2.0 * np.sin(x) + 0.01 * self.sin_lin(x)

    def lin_sin_quad(self, x):
        return self.lin_sin(x) - 0.03 * x * x

    def safety_function(self, x):
        exponent = 2.0
        intercept = 1.0
        border = 12.0
        return -1 * intercept * np.power((x / border), exponent) + intercept

    def safety_function_hard(self, x):
        exponent = 2.0
        intercept = 5.0
        border = 12.0
        return -1 * intercept * np.power((x / border), exponent) + intercept

    def eval(self, a, b, n):
        xs = np.linspace(a, b, n)
        fs = self.f(xs)
        return xs, fs

    def produce_data(self, a, b, n):
        xs = np.random.uniform(a, b, n)
        epsilon = np.random.normal(0, self.noise_level, n)
        ys = self.f(xs) + epsilon
        return xs, ys

    def produce_data_two_intervals(self, a1, b1, a2, b2, n1, n2):
        xs1 = np.random.uniform(a1, b1, n1)
        xs2 = np.random.uniform(a2, b2, n2)
        xs = np.concatenate((xs1, xs2), axis=0)
        n = n1 + n2
        epsilon = np.random.normal(0, self.noise_level, n)
        ys = self.f(xs) + epsilon
        return xs, ys

    def plot_func_with_generated_data(self, func_a, func_b, func_n, data_a, data_b, data_n):
        x_data, y_data = self.produce_data(data_a, data_b, data_n)
        func_x, func_y = self.eval(func_a, func_b, func_n)
        fig, ax = plt.subplots(1, 1)
        ax.plot(func_x, func_y)
        ax.plot(x_data, y_data, "ro")
        ax.grid()
        plt.show()


if __name__ == "__main__":
    func1 = GroundTruthFunction(FunctionType.SIN_LIN, 0.3)
    func1.plot_func_with_generated_data(0, 10, 100, 3, 5, 30)
