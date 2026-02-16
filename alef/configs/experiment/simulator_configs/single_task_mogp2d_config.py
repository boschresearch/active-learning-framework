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


class SingleTaskMOGP2DBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = False
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "single_task_mogp"


class SingleTaskMOGP2D0Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d0"
    name: str = "single_task_mogp2d0"


class SingleTaskMOGP2D1Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d1"
    name: str = "single_task_mogp2d1"


class SingleTaskMOGP2D2Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d2"
    name: str = "single_task_mogp2d2"


class SingleTaskMOGP2D3Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d3"
    name: str = "single_task_mogp2d3"


class SingleTaskMOGP2D4Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d4"
    name: str = "single_task_mogp2d4"


class SingleTaskMOGP2D5Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d5"
    name: str = "single_task_mogp2d5"


class SingleTaskMOGP2D6Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d6"
    name: str = "single_task_mogp2d6"


class SingleTaskMOGP2D7Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d7"
    name: str = "single_task_mogp2d7"


class SingleTaskMOGP2D8Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d8"
    name: str = "single_task_mogp2d8"


class SingleTaskMOGP2D9Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d9"
    name: str = "single_task_mogp2d9"


class SingleTaskMOGP2D10Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d10"
    name: str = "single_task_mogp2d10"


class SingleTaskMOGP2D11Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d11"
    name: str = "single_task_mogp2d11"


class SingleTaskMOGP2D12Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d12"
    name: str = "single_task_mogp2d12"


class SingleTaskMOGP2D13Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d13"
    name: str = "single_task_mogp2d13"


class SingleTaskMOGP2D14Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d14"
    name: str = "single_task_mogp2d14"


class SingleTaskMOGP2D15Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d15"
    name: str = "single_task_mogp2d15"


class SingleTaskMOGP2D16Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d16"
    name: str = "single_task_mogp2d16"


class SingleTaskMOGP2D17Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d17"
    name: str = "single_task_mogp2d17"


class SingleTaskMOGP2D18Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d18"
    name: str = "single_task_mogp2d18"


class SingleTaskMOGP2D19Config(SingleTaskMOGP2DBaseConfig):
    oracle_type: str = "mogp2d19"
    name: str = "single_task_mogp2d19"
