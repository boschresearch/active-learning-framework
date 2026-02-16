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


class SingleTaskMOGP1DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float = 0.01
    seed: int = 1234
    name: str = "single_task_mogp"


class SingleTaskMOGP1Dz0Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz0"
    name: str = "single_task_mogp1dz0"


class SingleTaskMOGP1Dz1Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz1"
    name: str = "single_task_mogp1dz1"


class SingleTaskMOGP1Dz2Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz2"
    name: str = "single_task_mogp1dz2"


class SingleTaskMOGP1Dz3Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz3"
    name: str = "single_task_mogp1dz3"


class SingleTaskMOGP1Dz4Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz4"
    name: str = "single_task_mogp1dz4"


class SingleTaskMOGP1Dz5Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz5"
    name: str = "single_task_mogp1dz5"


class SingleTaskMOGP1Dz6Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz6"
    name: str = "single_task_mogp1dz6"


class SingleTaskMOGP1Dz7Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz7"
    name: str = "single_task_mogp1dz7"


class SingleTaskMOGP1Dz8Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz8"
    name: str = "single_task_mogp1dz8"


class SingleTaskMOGP1Dz9Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz9"
    name: str = "single_task_mogp1dz9"


class SingleTaskMOGP1Dz10Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz10"
    name: str = "single_task_mogp1dz10"


class SingleTaskMOGP1Dz11Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz11"
    name: str = "single_task_mogp1dz11"


class SingleTaskMOGP1Dz12Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz12"
    name: str = "single_task_mogp1dz12"


class SingleTaskMOGP1Dz13Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz13"
    name: str = "single_task_mogp1dz13"


class SingleTaskMOGP1Dz14Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz14"
    name: str = "single_task_mogp1dz14"


class SingleTaskMOGP1Dz15Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz15"
    name: str = "single_task_mogp1dz15"


class SingleTaskMOGP1Dz16Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz16"
    name: str = "single_task_mogp1dz16"


class SingleTaskMOGP1Dz17Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz17"
    name: str = "single_task_mogp1dz17"


class SingleTaskMOGP1Dz18Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz18"
    name: str = "single_task_mogp1dz18"


class SingleTaskMOGP1Dz19Config(SingleTaskMOGP1DzBaseConfig):
    oracle_type: str = "mogp1dz19"
    name: str = "single_task_mogp1dz19"
