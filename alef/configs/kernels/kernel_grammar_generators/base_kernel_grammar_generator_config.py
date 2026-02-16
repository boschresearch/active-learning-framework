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

from pydantic import BaseSettings


class BaseKernelGrammarGeneratorConfig(BaseSettings):
    input_dimension: int
    name: str
    n_initial_factor_trailing: int = 5
    n_initial_flat_trailing: int = 200
    depth_initial_flat_trailing: int = 2
    do_flat_initial_trailing: bool = True
    n_exploration_trailing: int = 15
    exploration_p_geometric: float = 1.0 / 3.0
    n_exploitation_trailing: int = 50
    walk_length_exploitation_trailing: int = 5
    do_random_walk_exploitation_trailing: bool = False
    limit_n_additional_current_best: bool = True
