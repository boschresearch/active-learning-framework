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

import numpy as np
from alef.kernels.base_elementary_kernel import BaseElementaryKernel


class StationaryKernelGPflow:
    @property
    def prior_scale(self):
        assert isinstance(self, BaseElementaryKernel)
        D = self.get_input_dimension()
        dumpy_point = np.zeros([1, D])

        var_scale = self(dumpy_point, full_cov=False).numpy().reshape(-1)[0]
        std_scale = np.sqrt(var_scale)
        return std_scale
