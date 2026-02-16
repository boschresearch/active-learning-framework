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


class TransferTaskMOGP2DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "transfer_task_mogp"


class TransferTaskMOGP2Dz0Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz0"
    name: str = "transfer_task_mogp2dz0"


class TransferTaskMOGP2Dz1Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz1"
    name: str = "transfer_task_mogp2dz1"


class TransferTaskMOGP2Dz2Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz2"
    name: str = "transfer_task_mogp2dz2"


class TransferTaskMOGP2Dz3Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz3"
    name: str = "transfer_task_mogp2dz3"


class TransferTaskMOGP2Dz4Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz4"
    name: str = "transfer_task_mogp2dz4"


class TransferTaskMOGP2Dz5Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz5"
    name: str = "transfer_task_mogp2dz5"


class TransferTaskMOGP2Dz6Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz6"
    name: str = "transfer_task_mogp2dz6"


class TransferTaskMOGP2Dz7Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz7"
    name: str = "transfer_task_mogp2dz7"


class TransferTaskMOGP2Dz8Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz8"
    name: str = "transfer_task_mogp2dz8"


class TransferTaskMOGP2Dz9Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz9"
    name: str = "transfer_task_mogp2dz9"


class TransferTaskMOGP2Dz10Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz10"
    name: str = "transfer_task_mogp2dz10"


class TransferTaskMOGP2Dz11Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz11"
    name: str = "transfer_task_mogp2dz11"


class TransferTaskMOGP2Dz12Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz12"
    name: str = "transfer_task_mogp2dz12"


class TransferTaskMOGP2Dz13Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz13"
    name: str = "transfer_task_mogp2dz13"


class TransferTaskMOGP2Dz14Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz14"
    name: str = "transfer_task_mogp2dz14"


class TransferTaskMOGP2Dz15Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz15"
    name: str = "transfer_task_mogp2dz15"


class TransferTaskMOGP2Dz16Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz16"
    name: str = "transfer_task_mogp2dz16"


class TransferTaskMOGP2Dz17Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz17"
    name: str = "transfer_task_mogp2dz17"


class TransferTaskMOGP2Dz18Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz18"
    name: str = "transfer_task_mogp2dz18"


class TransferTaskMOGP2Dz19Config(TransferTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz19"
    name: str = "transfer_task_mogp2dz19"
