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

import numpy as np
import tensorflow as tf
import torch
from typing import Union
from abc import ABC, abstractmethod

from alef.enums.environment_enums import GPFramework


class BaseDistribution(ABC):
    """Base class for GP samplers."""

    def __init__(self, gp_framework: GPFramework):
        self.gp_framework = GPFramework

    @abstractmethod
    def draw_parameter(self, draw_hyper_prior: bool = False, draw_noise: bool = False):
        """
        draw hyper-priors, f, or noise of y|f

        arguments:

        draw_hyper_prior: whether to draw parameters from hyper-priors
        draw_noise: whether to draw the noise parameter as well (noise of y|f)

        """
        raise NotImplementedError

    @abstractmethod
    def f_sampler(self, x_data: Union[tf.Tensor, torch.Tensor]):
        """
        compute GP( mean(x_data), kernel(x_data) ), return in raw tf or torch type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        raise NotImplementedError

    @abstractmethod
    def y_sampler(self, x_data: Union[tf.Tensor, torch.Tensor]):
        """
        compute GP( mean(x_data), kernel(x_data) ) + noise_dist, return in raw tf or torch type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        raise NotImplementedError

    @abstractmethod
    def sample_f(self, x_data: np.ndarray):
        """
        sample f from GP( mean(x_data), kernel(x_data) )

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        raise NotImplementedError

    @abstractmethod
    def sample_y(self, x_data: np.ndarray):
        """
        sample y from GP( mean(x_data), kernel(x_data) ) + noise_dist(x_data)

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        raise NotImplementedError
