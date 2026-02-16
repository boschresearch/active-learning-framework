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

from typing import Union, Sequence
from alef.configs.experiment.simulator_configs.base_simulator_config import BaseSimulatorConfig
from alef.enums.simulator_enums import InitialDataGenerationMethod


class SingleTaskEngineInterpolatedBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = True
    data_set: str = "engine_oracle2"
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.GENERATE
    target_box_a = [-0.5, 0.5, -2.5, -2.5]
    target_box_width = [1.0, 1.0, 5, 5]
    input_idx: Sequence[Union[int, bool]] = [0, 1, 2, 3]
    output_idx: Sequence[Union[int, bool]] = [0]
    safety_idx: Sequence[Union[int, bool]] = [1]
    seed: int = 1234
    name: str = "single_task_engine_interpolated"


class SingleTaskEngineInterpolated_be_Config(SingleTaskEngineInterpolatedBaseConfig):
    output_idx: Sequence[Union[int, bool]] = [0]
    name: str = "single_task_engine_interpolated_be"


class SingleTaskEngineInterpolated_TEx_Config(SingleTaskEngineInterpolatedBaseConfig):
    output_idx: Sequence[Union[int, bool]] = [1]
    name: str = "single_task_engine_interpolated_TEx"


class SingleTaskEngineInterpolated_PI0v_Config(SingleTaskEngineInterpolatedBaseConfig):
    output_idx: Sequence[Union[int, bool]] = [3]
    name: str = "single_task_engine_interpolated_PI0v"


class SingleTaskEngineInterpolated_PI0s_Config(SingleTaskEngineInterpolatedBaseConfig):
    output_idx: Sequence[Union[int, bool]] = [4]
    name: str = "single_task_engine_interpolated_PI0s"


class SingleTaskEngineInterpolated_HC_Config(SingleTaskEngineInterpolatedBaseConfig):
    output_idx: Sequence[Union[int, bool]] = [5]
    name: str = "single_task_engine_interpolated_HC"


class SingleTaskEngineInterpolated_NOx_Config(SingleTaskEngineInterpolatedBaseConfig):
    output_idx: Sequence[Union[int, bool]] = [6]
    name: str = "single_task_engine_interpolated_NOx"
