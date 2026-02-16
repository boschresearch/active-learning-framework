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


class SingleTaskBraninConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = False
    oracle_type: str = "branin0"

    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE

    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "single_task_branin"


class SingleTaskBranin0Config(SingleTaskBraninConfig):
    oracle_type: str = "branin0"  # only affect initial data in this case
    name: str = "single_task_branin0"


class SingleTaskBranin1Config(SingleTaskBraninConfig):
    oracle_type: str = "branin1"  # only affect initial data in this case
    name: str = "single_task_branin1"


class SingleTaskBranin2Config(SingleTaskBraninConfig):
    oracle_type: str = "branin2"  # only affect initial data in this case
    name: str = "single_task_branin2"


class SingleTaskBranin3Config(SingleTaskBraninConfig):
    oracle_type: str = "branin3"  # only affect initial data in this case
    name: str = "single_task_branin3"


class SingleTaskBranin4Config(SingleTaskBraninConfig):
    oracle_type: str = "branin4"  # only affect initial data in this case
    name: str = "single_task_branin4"
