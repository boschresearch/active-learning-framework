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

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class BaseFeatureExtractor(ABC):
    @abstractmethod
    def forward(self, X: np.array) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_output_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_input_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def regularization_loss(self, x_data: np.array) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_mode(self, training_mode: bool):
        raise NotImplementedError
