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

import torch
from torch import nn
from abc import abstractmethod


class BaseSimulatedProcess(nn.Module):
    def __init__(self, design_net: nn.Module):
        super().__init__()
        self.design_net = design_net

    @property
    def input_domain(self):
        return self.design_net.input_domain

    def set_device(self, device: torch.device):
        self.to(device)

    @abstractmethod
    def process(self):
        """
        simulated experiment process for policy training.
        """
        raise NotImplementedError

    @abstractmethod
    def validation(self):
        """
        simulated experiment process for policy evaluation.
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.process(*args, **kwargs)


if __name__ == "__main__":
    print(BaseSimulatedProcess)
    print(BaseSimulatedProcess.__class__)
    print(isinstance(BaseSimulatedProcess, BaseSimulatedProcess.__class__))
