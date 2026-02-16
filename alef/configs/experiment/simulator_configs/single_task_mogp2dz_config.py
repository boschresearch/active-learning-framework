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


class SingleTaskMOGP2DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "single_task_mogp"


class SingleTaskMOGP2Dz0Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz0"
    name: str = "single_task_mogp2dz0"


class SingleTaskMOGP2Dz1Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz1"
    name: str = "single_task_mogp2dz1"


class SingleTaskMOGP2Dz2Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz2"
    name: str = "single_task_mogp2dz2"


class SingleTaskMOGP2Dz3Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz3"
    name: str = "single_task_mogp2dz3"


class SingleTaskMOGP2Dz4Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz4"
    name: str = "single_task_mogp2dz4"


class SingleTaskMOGP2Dz5Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz5"
    name: str = "single_task_mogp2dz5"


class SingleTaskMOGP2Dz6Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz6"
    name: str = "single_task_mogp2dz6"


class SingleTaskMOGP2Dz7Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz7"
    name: str = "single_task_mogp2dz7"


class SingleTaskMOGP2Dz8Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz8"
    name: str = "single_task_mogp2dz8"


class SingleTaskMOGP2Dz9Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz9"
    name: str = "single_task_mogp2dz9"


class SingleTaskMOGP2Dz10Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz10"
    name: str = "single_task_mogp2dz10"


class SingleTaskMOGP2Dz11Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz11"
    name: str = "single_task_mogp2dz11"


class SingleTaskMOGP2Dz12Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz12"
    name: str = "single_task_mogp2dz12"


class SingleTaskMOGP2Dz13Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz13"
    name: str = "single_task_mogp2dz13"


class SingleTaskMOGP2Dz14Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz14"
    name: str = "single_task_mogp2dz14"


class SingleTaskMOGP2Dz15Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz15"
    name: str = "single_task_mogp2dz15"


class SingleTaskMOGP2Dz16Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz16"
    name: str = "single_task_mogp2dz16"


class SingleTaskMOGP2Dz17Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz17"
    name: str = "single_task_mogp2dz17"


class SingleTaskMOGP2Dz18Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz18"
    name: str = "single_task_mogp2dz18"


class SingleTaskMOGP2Dz19Config(SingleTaskMOGP2DzBaseConfig):
    oracle_type: str = "mogp2dz19"
    name: str = "single_task_mogp2dz19"
