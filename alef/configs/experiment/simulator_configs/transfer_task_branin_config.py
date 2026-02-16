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


class TransferTaskBraninBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = False
    oracle_type: str = "branin"
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "transfer_task_branin"


class TransferTaskBranin0Config(TransferTaskBraninBaseConfig):
    oracle_type: str = "branin0"
    name: str = "transfer_task_branin0"


class TransferTaskBranin1Config(TransferTaskBraninBaseConfig):
    oracle_type: str = "branin1"
    name: str = "transfer_task_branin1"


class TransferTaskBranin2Config(TransferTaskBraninBaseConfig):
    oracle_type: str = "branin2"
    name: str = "transfer_task_branin2"


class TransferTaskBranin3Config(TransferTaskBraninBaseConfig):
    oracle_type: str = "branin3"
    name: str = "transfer_task_branin3"


class TransferTaskBranin4Config(TransferTaskBraninBaseConfig):
    oracle_type: str = "branin4"
    name: str = "transfer_task_branin4"
