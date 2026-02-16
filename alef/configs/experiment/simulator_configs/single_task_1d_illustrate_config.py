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

from alef.configs.experiment.simulator_configs.base_simulator_config import BaseSimulatorConfig
from alef.enums.simulator_enums import InitialDataGenerationMethod


class SingleTaskIllustrateConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = False
    oracle_type: str = "illustrate"
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.GENERATE
    target_box_a = 0.0
    target_box_width = 0.25
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "single_task_illustrate"
