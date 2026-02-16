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

from abc import abstractmethod, abstractproperty
from typing import List
from .base_loss import BaseLoss
import numpy as np

# from alef.configs.active_learner.amortized_policies.loss_configs import BasicAmortizedPolicyLossConfig


class BaseCurriculum(BaseLoss):
    def set_current_loss(self, loss: BaseLoss):
        self._current_loss = loss

    @property
    def current_loss(self):
        return self._current_loss

    @abstractproperty
    def epoch_idx(self):
        r"""
        return the current epoch index
        """
        raise NotImplementedError

    @abstractproperty
    def step_per_epoch_idx(self):
        r"""
        return the current step index in the current epoch
        """
        raise NotImplementedError

    @abstractproperty
    def step_idx(self):
        r"""
        return the current step index (accross all epochs)
        """
        raise NotImplementedError

    @abstractproperty
    def num_steps(self):
        r"""
        return the total number of steps
        """
        raise NotImplementedError

    @abstractmethod
    def differentiable_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def loss(self, *args, **kwargs):
        pass

    def validation(self, test_process, *args, **kwargs):
        return self.current_loss.validation(test_process, *args, **kwargs)


class TrivialLossCurriculum(BaseCurriculum):
    def __init__(
        self,
        loss: BaseLoss,
        num_epochs: int,
        epochs_size: int,
        name: str = None,
    ):
        assert num_epochs > 0
        assert epochs_size > 0
        self.name = name
        self._num_epochs = num_epochs
        self._epochs_size = epochs_size
        self.set_current_loss(loss)

        self._epoch_idx = 0
        self._step_per_epoch_idx = 0
        self._step_idx = 0

    @property
    def epoch_idx(self):
        r"""
        return the current epoch index
        """
        return self._epoch_idx

    @property
    def step_per_epoch_idx(self):
        r"""
        return the current step index in the current epoch
        """
        return self._step_per_epoch_idx

    @abstractproperty
    def step_idx(self):
        r"""
        return the current step index (accross all epochs)
        """
        return self._step_idx

    @property
    def num_steps(self):
        r"""
        return the total number of steps
        """
        return self._num_epochs * self._epochs_size

    def differentiable_loss(self, *args, **kwargs):
        dl = self.current_loss.differentiable_loss(*args, **kwargs)
        self._step_idx += 1
        self._step_per_epoch_idx += 1
        if self._step_per_epoch_idx == self._epochs_size:
            self._step_per_epoch_idx = 0
            self._epoch_idx += 1
        return dl

    def loss(self, *args, **kwargs):
        return self.current_loss.loss(*args, **kwargs)


class LossCurriculum(BaseCurriculum):
    def __init__(
        self, loss_list: List[BaseLoss], num_epochs_list: List[int], epochs_size_list: List[int], name: str = None
    ):
        assert len(loss_list) == len(num_epochs_list) == len(epochs_size_list)
        self.name = name
        self._loss_list = loss_list
        self._num_epochs_list = num_epochs_list
        self._epochs_size_list = epochs_size_list

        self.set_current_loss(self._loss_list[0])
        self._current_epochs_size = self._epochs_size_list[0]
        self._switch_step_idx = np.cumsum(np.array(self._num_epochs_list) * np.array(self._epochs_size_list))[
            :-1
        ].tolist()
        self._num_step = (np.array(self._num_epochs_list) * np.array(self._epochs_size_list)).sum()

        self._epoch_idx = 0
        self._step_per_epoch_idx = 0
        self._step_idx = 0

    @property
    def epoch_idx(self):
        r"""
        return the current epoch index
        """
        return self._epoch_idx

    @property
    def step_per_epoch_idx(self):
        r"""
        return the current step index in the current epoch
        """
        return self._step_per_epoch_idx

    @abstractproperty
    def step_idx(self):
        r"""
        return the current step index (accross all epochs)
        """
        return self._step_idx

    @property
    def num_steps(self):
        r"""
        return the total number of steps
        """
        return self._num_step

    def differentiable_loss(self, *args, **kwargs):
        if self._step_idx in self._switch_step_idx:
            print(f"Current loss: {self.current_loss.__class__.__name__}")
            self.set_current_loss(self._loss_list[self._switch_step_idx.index(self._step_idx) + 1])
            print(f"Switch to loss: {self.current_loss.__class__.__name__}")

        dl = self.current_loss.differentiable_loss(*args, **kwargs)

        self._step_idx += 1
        self._step_per_epoch_idx += 1
        if self._step_per_epoch_idx == self._current_epochs_size:
            self._step_per_epoch_idx = 0
            self._epoch_idx += 1

        return dl

    def loss(self, *args, **kwargs):
        return self.current_loss.loss(*args, **kwargs)
