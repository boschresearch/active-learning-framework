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


class TransferTaskMOGP1DBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = False
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "transfer_task_mogp"


class TransferTaskMOGP1D0Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d0"
    name: str = "transfer_task_mogp1d0"


class TransferTaskMOGP1D1Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d1"
    name: str = "transfer_task_mogp1d1"


class TransferTaskMOGP1D2Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d2"
    name: str = "transfer_task_mogp1d2"


class TransferTaskMOGP1D3Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d3"
    name: str = "transfer_task_mogp1d3"


class TransferTaskMOGP1D4Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d4"
    name: str = "transfer_task_mogp1d4"


class TransferTaskMOGP1D5Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d5"
    name: str = "transfer_task_mogp1d5"


class TransferTaskMOGP1D6Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d6"
    name: str = "transfer_task_mogp1d6"


class TransferTaskMOGP1D7Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d7"
    name: str = "transfer_task_mogp1d7"


class TransferTaskMOGP1D8Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d8"
    name: str = "transfer_task_mogp1d8"


class TransferTaskMOGP1D9Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d9"
    name: str = "transfer_task_mogp1d9"


class TransferTaskMOGP1D10Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d10"
    name: str = "transfer_task_mogp1d10"


class TransferTaskMOGP1D11Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d11"
    name: str = "transfer_task_mogp1d11"


class TransferTaskMOGP1D12Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d12"
    name: str = "transfer_task_mogp1d12"


class TransferTaskMOGP1D13Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d13"
    name: str = "transfer_task_mogp1d13"


class TransferTaskMOGP1D14Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d14"
    name: str = "transfer_task_mogp1d14"


class TransferTaskMOGP1D15Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d15"
    name: str = "transfer_task_mogp1d15"


class TransferTaskMOGP1D16Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d16"
    name: str = "transfer_task_mogp1d16"


class TransferTaskMOGP1D17Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d17"
    name: str = "transfer_task_mogp1d17"


class TransferTaskMOGP1D18Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d18"
    name: str = "transfer_task_mogp1d18"


class TransferTaskMOGP1D19Config(TransferTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d19"
    name: str = "transfer_task_mogp1d19"
