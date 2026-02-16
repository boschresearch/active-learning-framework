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

import torch
from abc import abstractmethod


class BaseSampler(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError  # we don't really need this

    def set_device(self, device: torch.device):
        self.to(device)
        for m in self.modules():
            m.to(device)

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
    def f_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ), return in raw torch type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        raise NotImplementedError

    @abstractmethod
    def y_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ) + noise_dist, return in raw torch type

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        """
        raise NotImplementedError
