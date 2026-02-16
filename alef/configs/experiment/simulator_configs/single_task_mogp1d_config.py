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


class SingleTaskMOGP1DBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = False
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "single_task_mogp"


class SingleTaskMOGP1D0Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d0"
    name: str = "single_task_mogp1d0"


class SingleTaskMOGP1D1Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d1"
    name: str = "single_task_mogp1d1"


class SingleTaskMOGP1D2Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d2"
    name: str = "single_task_mogp1d2"


class SingleTaskMOGP1D3Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d3"
    name: str = "single_task_mogp1d3"


class SingleTaskMOGP1D4Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d4"
    name: str = "single_task_mogp1d4"


class SingleTaskMOGP1D5Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d5"
    name: str = "single_task_mogp1d5"


class SingleTaskMOGP1D6Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d6"
    name: str = "single_task_mogp1d6"


class SingleTaskMOGP1D7Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d7"
    name: str = "single_task_mogp1d7"


class SingleTaskMOGP1D8Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d8"
    name: str = "single_task_mogp1d8"


class SingleTaskMOGP1D9Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d9"
    name: str = "single_task_mogp1d9"


class SingleTaskMOGP1D10Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d10"
    name: str = "single_task_mogp1d10"


class SingleTaskMOGP1D11Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d11"
    name: str = "single_task_mogp1d11"


class SingleTaskMOGP1D12Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d12"
    name: str = "single_task_mogp1d12"


class SingleTaskMOGP1D13Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d13"
    name: str = "single_task_mogp1d13"


class SingleTaskMOGP1D14Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d14"
    name: str = "single_task_mogp1d14"


class SingleTaskMOGP1D15Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d15"
    name: str = "single_task_mogp1d15"


class SingleTaskMOGP1D16Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d16"
    name: str = "single_task_mogp1d16"


class SingleTaskMOGP1D17Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d17"
    name: str = "single_task_mogp1d17"


class SingleTaskMOGP1D18Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d18"
    name: str = "single_task_mogp1d18"


class SingleTaskMOGP1D19Config(SingleTaskMOGP1DBaseConfig):
    oracle_type: str = "mogp1d19"
    name: str = "single_task_mogp1d19"
