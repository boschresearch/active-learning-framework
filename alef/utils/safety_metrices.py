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

from typing import Sequence
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from alef.utils.connected_component_labelling import TrivialDetector, TwoPassHighDim


class SafetyAreaMeasure:
    """
    this class helps safe learning methods (safe BO, safe AL) quantify safe lands
    how to use:
    1. initialize this class:
        safety_area_measure = SafetyAreaMeasure()
        safety_area_measure.set_object_detector(input_dimension)
    2. set ground true safety mask:
        safety_area_measure.true_safe_lands(grid, true_safety_mask)
    3. compute true positive of the prediction:
        safety_area_measure.true_positive_lands(mask)
    """

    def __init__(self):
        self.__safeland_area = 0
        self.__safeland_individual_area = []  # tuple (area of true land1, area of true land2, etc)
        self.__safeland_hit_area = []
        self.__safeland_individual_hit_area = []  # each iter, tuple (true positive area of land1, etc)
        self.__safeland_falsealarm_area = []
        self.__safeland_individual_falsealarm_area = []  # each iter, tuple (false positive area of land1, etc)

    def set_object_detector(self, grid_dimension: int):
        if grid_dimension == 1:
            self.__obj_detector = TrivialDetector()
        elif grid_dimension >= 2:
            self.__obj_detector = TwoPassHighDim()
        self.reset()

    def reset(self):
        self.__safeland_area = 0
        self.__safeland_individual_area = []  # tuple (area of true land1, area of true land2, etc)
        self.__safeland_hit_area = []
        self.__safeland_individual_hit_area = []  # each iter, tuple (true positive area of land1, etc)
        self.__safeland_falsealarm_area = []
        self.__safeland_individual_falsealarm_area = []  # each iter, tuple (false positive area of land1, etc)

    def _label_safe_lands(self, grid, mask):
        r"""
        grid: [N, D] array
        mask: [N, 1] array

        return
        num_labels: int
        labels: [N, 1] int array
        """
        return self.__obj_detector.cca(grid, mask)

    def true_safe_lands_from_file(self, label_grid_path: str, sheet_name: str):
        r"""
        label_grid_path: txt path to a [N, 1] array file
        """
        df = pd.read_excel(label_grid_path, sheet_name=sheet_name, header=[0], index_col=[0])
        grid = df.to_numpy().astype(float)[:-1]
        land_labels = df["safe_land"].to_numpy().astype(float)
        num_lands = len(np.unique(land_labels.reshape(-1))) - 1

        areas = [np.sum(land_labels == label) / land_labels.shape[0] for label in range(1, num_lands + 1)]

        self.__num_lands = num_lands
        self.__grid = grid
        self.__true_land_labels = land_labels
        self.__safeland_area = np.sum(land_labels > 0) / land_labels.shape[0]
        self.__safeland_individual_area.append(tuple(areas))

    def true_safe_lands(self, grid: np.ndarray, true_mask: np.ndarray):
        r"""
        grid: [N, D] array
        mask: [N, 1] array
        """
        num_lands, land_labels = self._label_safe_lands(grid, true_mask)
        land_labels = land_labels.astype(float)
        areas = [np.sum(land_labels == label) / land_labels.shape[0] for label in range(1, num_lands + 1)]

        self.__num_lands = num_lands
        self.__grid = grid
        self.__true_land_labels = land_labels
        self.__safeland_area = np.sum(land_labels > 0) / land_labels.shape[0]
        self.__safeland_individual_area.append(tuple(areas))

    @property
    def num_lands(self):
        return self.__num_lands

    @property
    def grid(self):
        return self.__grid

    @property
    def land_labels(self):
        return self.__true_land_labels

    @property
    def safeland_area(self):
        return self.__safeland_area

    @property
    def safeland_individual_area(self):
        return self.__safeland_individual_area

    @property
    def safeland_hit_area(self):
        return self.__safeland_hit_area

    @property
    def safeland_individual_hit_area(self):
        return self.__safeland_individual_hit_area

    @property
    def safeland_falsealarm_area(self):
        return self.__safeland_falsealarm_area

    @property
    def safeland_individual_falsealarm_area(self):
        return self.__safeland_individual_falsealarm_area

    def label_points(self, x):
        return self.__obj_detector.label_points(x, self.__grid, self.__true_land_labels)

    def true_positive_lands(self, mask):
        r"""
        mask: [N, 1] array

        please make sure that the mask use the same grid as when we call true_safe_lands
        """
        mask = np.reshape(mask, -1).astype(int)
        land_labels = self.__true_land_labels.astype(float).copy()
        land_labels[mask == 0] = 0
        areas = [np.sum(land_labels == label) / land_labels.shape[0] for label in range(1, self.__num_lands + 1)]

        self.__safeland_hit_area.append(np.sum(land_labels > 0) / land_labels.shape[0])
        self.__safeland_individual_hit_area.append(tuple(areas))

    def get_total_iter_num_true_positive(self):
        return len(self.__safeland_hit_area)

    def false_positive_lands(self, mask):
        r"""
        mask: [N, 1] array

        please make sure that the mask use the same grid as when we call true_safe_lands
        """
        fp_mask = mask.astype(float).copy()
        true_labels = self.__true_land_labels.reshape(-1)
        fp_mask[true_labels > 0] = 0

        self.__safeland_falsealarm_area.append(np.sum(fp_mask > 0) / fp_mask.shape[0])

        """
        For individual safe lands:
        The problem is in each iteration, we might have different number of predictive safe lands,
        so each tuple might turn out having different number of element.
        """
        # num_lands, land_labels = self.__label_safe_lands(self.__grid, mask)
        # land_labels[self.__true_land_labels.reshape(-1)>0] = 0
        # areas = [np.sum(land_labels == label) / land_labels.shape[0] for label in range(1, num_lands+1)]

        # self.__safeland_individual_falsealarm_area.append(tuple(areas))

    def get_total_iter_num_false_positive(self):
        return len(self.__safeland_falsealarm_area)

    def export_df(self, df_index):
        true_safe_area = np.array([self.__safeland_area])
        true_posi_area = np.reshape(self.__safeland_hit_area, [-1, 1])
        false_alarm_area = np.reshape(self.__safeland_falsealarm_area, [-1, 1])

        true_safe_indiv_area = np.hstack(self.__safeland_individual_area).reshape([1, self.__num_lands])
        true_posi_indiv_area = np.vstack(self.__safeland_individual_hit_area)

        df1 = pd.DataFrame(true_safe_area, columns=["true_safe_area_all"], index=df_index[[0]])
        df2 = pd.DataFrame(true_posi_area, columns=["true_posi_area_all"], index=df_index)
        df3 = pd.DataFrame(false_alarm_area, columns=["false_posi_area_all"], index=df_index)

        df4 = pd.DataFrame(
            true_safe_indiv_area,
            columns=[f"true_safe_area{i + 1}" for i in range(1, self.__num_lands + 1)],
            index=df_index[[0]],
        )
        df5 = pd.DataFrame(
            true_posi_indiv_area,
            columns=[f"true_posi_area{i + 1}" for i in range(1, self.__num_lands + 1)],
            index=df_index,
        )

        return pd.concat([df1, df2, df3, df4, df5], axis=1)

    def export_plot(
        self,
        x_ticks_true_positive: Sequence = None,
        x_ticks_false_alarm: Sequence = None,
        save_plot: bool = False,
        file_name: str = "",
        file_path: str = "",
    ):
        true_safe_area = self.__safeland_area
        true_posi_area = np.reshape(self.__safeland_hit_area, -1)
        false_alarm_area = np.reshape(self.__safeland_falsealarm_area, -1)
        if x_ticks_true_positive is None:
            x_ticks_true_positive = np.arange(len(true_posi_area))
        if x_ticks_false_alarm is None:
            x_ticks_false_alarm = np.arange(len(false_alarm_area))
        assert len(x_ticks_true_positive) == len(true_posi_area)
        assert len(x_ticks_false_alarm) == len(false_alarm_area)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
        axs.plot(
            [
                min(min(x_ticks_true_positive), min(x_ticks_false_alarm)),
                max(max(x_ticks_true_positive), max(x_ticks_false_alarm)),
            ],
            [true_safe_area, true_safe_area],
            color="black",
            linestyle="--",
            label="safe area, all space",
        )
        axs.plot(x_ticks_true_positive, true_posi_area, color="C0", label="hit area, all space")
        axs.plot(x_ticks_false_alarm, false_alarm_area, color="C1", label="false alarm area, all space")
        axs.legend()

        if save_plot:
            plt.savefig(os.path.join(file_path, file_name))
            plt.close()
        else:
            plt.show()
