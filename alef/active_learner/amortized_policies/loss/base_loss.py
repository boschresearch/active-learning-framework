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

from abc import abstractmethod
from alef.active_learner.amortized_policies.simulated_processes.base_process import BaseSimulatedProcess


class BaseLoss(object):
    def __init__(self, batch_size, data_source=None, name: str = None):
        self.batch_size = batch_size
        self.data_source = data_source
        self.name = name

    @abstractmethod
    def differentiable_loss(self, process: BaseSimulatedProcess, *args, **kwargs):
        """
        compute the loss that can be differentiated with autograd

        return a loss value in torch tensor
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, process: BaseSimulatedProcess, *args, **kwargs):
        """
        compute the loss which does not need to be differentiated (for visualization, analysis etc)

        return a loss value in torch tensor
        """
        raise NotImplementedError

    @abstractmethod
    def validation(self, process: BaseSimulatedProcess, *args, **kwargs):
        """
        compute the validation values which does not need to be differentiated (for visualization, analysis etc)

        return tuple of 2 floats, the mean and standard error of the validation values
        """
        raise NotImplementedError
