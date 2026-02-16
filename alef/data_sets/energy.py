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

import pandas as pd
import numpy as np
from enum import Enum
from alef.utils.plot_utils import active_learning_nd_plot
from alef.enums.data_sets_enums import InputPreprocessingType, OutputPreprocessingType
from alef.utils.utils import normalize_data, min_max_normalize_data
from alef.data_sets.base_data_set import StandardDataSet
import os


class OutputType(Enum):
    Y1 = 1
    Y2 = 2


class Energy(StandardDataSet):
    def __init__(self, base_path, file_name="energy_data.csv"):
        super().__init__()
        self.file_path = os.path.join(base_path, file_name)
        self.input_preprocessing_type = InputPreprocessingType.MIN_MAX_NORMALIZATION
        self.output_preprocessing_type = OutputPreprocessingType.NORMALIZATION
        self.output_type = OutputType.Y1
        if self.output_type == OutputType.Y1:
            self.name = "Energy1"
        elif self.output_type == OutputType.Y2:
            self.name = "Energy2"

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=",")
        x_list = []
        label_name_y1 = "Y1"
        label_name_y2 = "Y2"
        if self.output_type == OutputType.Y1:
            label_name = label_name_y1
        elif self.output_type == OutputType.Y2:
            label_name = label_name_y2
        print(df.columns)
        assert label_name in df.columns
        for col_name in df.columns:
            if not (col_name == label_name_y1 or col_name == label_name_y2) and len(col_name) == 2:
                x_list.append(np.expand_dims(df[col_name].to_numpy(), axis=1))
        y = np.expand_dims(df[label_name].to_numpy(), axis=1)
        self.length = y.shape[0]
        self.x = np.concatenate(x_list, axis=1)
        self.y = y
        if self.input_preprocessing_type == InputPreprocessingType.NORMALIZATION:
            self.x = normalize_data(self.x)
        elif self.input_preprocessing_type == InputPreprocessingType.MIN_MAX_NORMALIZATION:
            self.x = min_max_normalize_data(self.x)

        if self.output_preprocessing_type == OutputPreprocessingType.NORMALIZATION:
            self.y = normalize_data(self.y)
        print(self.x.shape)
        print(self.y.shape)

    def plot_scatter(self):
        xs, ys = self.sample(300)
        active_learning_nd_plot(xs, ys)
