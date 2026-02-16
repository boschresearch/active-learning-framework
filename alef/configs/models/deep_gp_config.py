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

from alef.configs.models.base_model_config import BaseModelConfig


class DeepGPConfig(BaseModelConfig):
    n_iter: int = 3000
    n_layer: int = 2
    max_n_inducing_points: int = 300
    learning_rate: float = 0.01
    initial_likelihood_noise_variance: float = 0.01
    name = "DeepGP"


class ThreeLayerDeepGPConfig(DeepGPConfig):
    n_layer: int = 3
    name = "ThreeLayerDeepGP"


class FiveLayerDeepGPConfig(DeepGPConfig):
    n_layer: int = 5
    name = "FiveLayerDeepGP"
