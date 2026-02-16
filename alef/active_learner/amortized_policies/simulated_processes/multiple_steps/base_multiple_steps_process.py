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


from torch import nn
from alef.active_learner.amortized_policies.simulated_processes.base_process import BaseSimulatedProcess


class BaseMultipleStepsProcess(BaseSimulatedProcess):
    def __init__(self, design_net: nn.Module, n_steps: int):
        super().__init__(design_net=design_net)
        self.n_steps = n_steps  # number of experiments
