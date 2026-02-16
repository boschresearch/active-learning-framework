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

from alef.utils.plot_utils import active_learning_nd_plot
from alef.enums.data_sets_enums import InputPreprocessingType, OutputPreprocessingType
from alef.utils.utils import normalize_data, min_max_normalize_data
from alef.data_sets.base_data_set import StandardDataSet
import os


class Yacht(StandardDataSet):
    def __init__(self, base_path, file_name="yacht_hydrodynamics.data"):
        super().__init__()
        self.file_path = os.path.join(base_path, file_name)
        self.input_preprocessing_type = InputPreprocessingType.MIN_MAX_NORMALIZATION
        self.output_preprocessing_type = OutputPreprocessingType.NORMALIZATION
        self.name = "Yacht"

    def load_data_set(self):
        data = np.loadtxt(self.file_path)
        x_lon = np.expand_dims(data[:, 0], axis=1)
        x_pris = np.expand_dims(data[:, 1], axis=1)
        x_displace = np.expand_dims(data[:, 2], axis=1)
        x_beam_draught = np.expand_dims(data[:, 3], axis=1)
        x_length_beam = np.expand_dims(data[:, 4], axis=1)
        x_froude = np.expand_dims(data[:, 5], axis=1)
        y_resist = np.expand_dims(data[:, 6], axis=1)
        self.x = np.concatenate((x_lon, x_pris, x_displace, x_beam_draught, x_length_beam, x_froude), axis=1)
        self.y = y_resist
        self.length = len(self.y)
        print(self.length)
        if self.input_preprocessing_type == InputPreprocessingType.NORMALIZATION:
            self.x = normalize_data(self.x)
        elif self.input_preprocessing_type == InputPreprocessingType.MIN_MAX_NORMALIZATION:
            self.x = min_max_normalize_data(self.x)

        if self.output_preprocessing_type == OutputPreprocessingType.NORMALIZATION:
            self.y = normalize_data(self.y)
        print(self.x)

    def plot_scatter(self):
        xs, ys = self.sample(300)
        active_learning_nd_plot(xs, ys)
