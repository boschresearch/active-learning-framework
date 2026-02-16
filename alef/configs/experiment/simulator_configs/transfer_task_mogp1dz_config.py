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


class TransferTaskMOGP1DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "transfer_task_mogp"


class TransferTaskMOGP1Dz0Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz0"
    name: str = "transfer_task_mogp1dz0"


class TransferTaskMOGP1Dz1Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz1"
    name: str = "transfer_task_mogp1dz1"


class TransferTaskMOGP1Dz2Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz2"
    name: str = "transfer_task_mogp1dz2"


class TransferTaskMOGP1Dz3Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz3"
    name: str = "transfer_task_mogp1dz3"


class TransferTaskMOGP1Dz4Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz4"
    name: str = "transfer_task_mogp1dz4"


class TransferTaskMOGP1Dz5Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz5"
    name: str = "transfer_task_mogp1dz5"


class TransferTaskMOGP1Dz6Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz6"
    name: str = "transfer_task_mogp1dz6"


class TransferTaskMOGP1Dz7Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz7"
    name: str = "transfer_task_mogp1dz7"


class TransferTaskMOGP1Dz8Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz8"
    name: str = "transfer_task_mogp1dz8"


class TransferTaskMOGP1Dz9Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz9"
    name: str = "transfer_task_mogp1dz9"


class TransferTaskMOGP1Dz10Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz10"
    name: str = "transfer_task_mogp1dz10"


class TransferTaskMOGP1Dz11Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz11"
    name: str = "transfer_task_mogp1dz11"


class TransferTaskMOGP1Dz12Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz12"
    name: str = "transfer_task_mogp1dz12"


class TransferTaskMOGP1Dz13Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz13"
    name: str = "transfer_task_mogp1dz13"


class TransferTaskMOGP1Dz14Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz14"
    name: str = "transfer_task_mogp1dz14"


class TransferTaskMOGP1Dz15Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz15"
    name: str = "transfer_task_mogp1dz15"


class TransferTaskMOGP1Dz16Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz16"
    name: str = "transfer_task_mogp1dz16"


class TransferTaskMOGP1Dz17Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz17"
    name: str = "transfer_task_mogp1dz17"


class TransferTaskMOGP1Dz18Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz18"
    name: str = "transfer_task_mogp1dz18"


class TransferTaskMOGP1Dz19Config(TransferTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz19"
    name: str = "transfer_task_mogp1dz19"
