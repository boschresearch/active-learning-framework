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


class TransferTaskMOGP2DBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = False
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "transfer_task_mogp"


class TransferTaskMOGP2D0Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d0"
    name: str = "transfer_task_mogp2d0"


class TransferTaskMOGP2D1Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d1"
    name: str = "transfer_task_mogp2d1"


class TransferTaskMOGP2D2Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d2"
    name: str = "transfer_task_mogp2d2"


class TransferTaskMOGP2D3Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d3"
    name: str = "transfer_task_mogp2d3"


class TransferTaskMOGP2D4Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d4"
    name: str = "transfer_task_mogp2d4"


class TransferTaskMOGP2D5Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d5"
    name: str = "transfer_task_mogp2d5"


class TransferTaskMOGP2D6Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d6"
    name: str = "transfer_task_mogp2d6"


class TransferTaskMOGP2D7Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d7"
    name: str = "transfer_task_mogp2d7"


class TransferTaskMOGP2D8Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d8"
    name: str = "transfer_task_mogp2d8"


class TransferTaskMOGP2D9Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d9"
    name: str = "transfer_task_mogp2d9"


class TransferTaskMOGP2D10Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d10"
    name: str = "transfer_task_mogp2d10"


class TransferTaskMOGP2D11Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d11"
    name: str = "transfer_task_mogp2d11"


class TransferTaskMOGP2D12Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d12"
    name: str = "transfer_task_mogp2d12"


class TransferTaskMOGP2D13Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d13"
    name: str = "transfer_task_mogp2d13"


class TransferTaskMOGP2D14Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d14"
    name: str = "transfer_task_mogp2d14"


class TransferTaskMOGP2D15Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d15"
    name: str = "transfer_task_mogp2d15"


class TransferTaskMOGP2D16Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d16"
    name: str = "transfer_task_mogp2d16"


class TransferTaskMOGP2D17Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d17"
    name: str = "transfer_task_mogp2d17"


class TransferTaskMOGP2D18Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d18"
    name: str = "transfer_task_mogp2d18"


class TransferTaskMOGP2D19Config(TransferTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d19"
    name: str = "transfer_task_mogp2d19"
