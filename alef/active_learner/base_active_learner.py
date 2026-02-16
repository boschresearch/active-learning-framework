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

import os
import abc
import numpy as np
import pandas as pd
from typing import List
from alef.models.base_model import BaseModel
from alef.enums.active_learner_enums import ValidationType


class BaseActiveLearner(abc.ABC):
    def __init__(self):
        super().__init__()
        self.validation_metrics: pd.DataFrame = None
        self.validation_type: ValidationType = None
        self.validation_at: List[int] = None
        self.data_history: pd.DataFrame = None
        self.ground_truth_available: bool = False
        self.do_plotting: bool = False
        self.save_plots: bool = False
        self.plot_path: str = ""
        self.save_result: bool = False
        self.summary_path: str = ""

    def initialize_validation_metrics(self):
        """
        initializes the validation metrics dataframe
        """
        self.__validation_singleoutput = True
        self.validation_metrics = pd.DataFrame(columns=["step_index", self.validation_type.name])

    def initialize_multioutput_validation_metrics(self, output_dimension: int):
        """
        initializes the validation metrics dataframe
        """
        self.__validation_singleoutput = False
        self.validation_metrics = pd.DataFrame(
            columns=[
                ["step_index", *([self.validation_type.name] * output_dimension)],
                ["", *[f"output_{i}" for i in range(output_dimension)]],
            ]
        )

    def add_validation_value(self, idx: int, value: float):
        """
        adds a validation value to the validation metrics dataframe
        """
        assert self.__validation_singleoutput
        self.validation_metrics.loc[len(self.validation_metrics), :] = [idx, value]

    def add_multioutput_validation_array(self, idx: int, values: np.ndarray):
        """
        adds a validation value to the validation metrics dataframe
        """
        assert not self.__validation_singleoutput
        self.validation_metrics.loc[len(self.validation_metrics), :] = [idx, *values]

    def set_model(self, model: BaseModel):
        """
        sets the model that is used for active learning
        """
        self.model = model

    def set_do_plotting(self, do_plotting: bool):
        """
        Method for specifying if plotting should be done

        Arguments:
            do_plotting - bool : flag if plotting should be done
        """
        self.do_plotting = do_plotting

    def save_plots_to_path(self, plot_path: str):
        """
        method to specify that plots are saved to a path
        """
        self.save_plots = True
        self.plot_path = plot_path

    def save_experiment_summary_to_path(self, summary_path: str, filename="AL_result.xlsx"):
        """
        method to specify that experiment summary is saved to a path
        """
        self.save_result = True
        self.summary_path = summary_path
        self.summary_filename = filename

    def set_test_set(self, x_test: np.array, y_test: np.array):
        """
        Method for setting the test set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n test datapoints with dimension d
        y_train - np.array - array of shape [n,1] containing the outputs of n test datapoints
        """
        self.x_test = x_test
        self.y_test = y_test

    def set_train_set(self, x_train: np.array, y_train: np.array):
        """
        Method for setting the train set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n training datapoints with dimension d
        y_train - np.array - array of shape [n,1] containing the outputs of n training datapoints
        """
        self.x_data = x_train
        self.y_data = y_train
        self._initialize_data_history()

    def add_train_point(self, idx, query, new_y):
        """
        Method for adding a single datapoint to the training set
        Arguments:
        idx - int - index of the iteration
        query - np.array - array of shape [d] containing the input of the datapoint with dimension d
        new_y - float - the output of the datapoint
        """
        self.x_data = np.vstack((self.x_data, query))
        self.y_data = np.vstack((self.y_data, [new_y]))
        self.data_history.loc[len(self.data_history), :] = [idx, *query, new_y]

    def add_batch_train_point(self, idx, queries, new_y):
        """
        Method for adding a single datapoint to the training set
        Arguments:
        idx - int - index of the iteration
        queries - np.array - array of shape [n, d] containing the input of the datapoint with dimension d
        new_y - np.array - array of shape [n, 1] containing the output of the datapoint
        """
        self.x_data = np.vstack((self.x_data, queries))
        self.y_data = np.vstack((self.y_data, new_y))
        self.data_history = self.data_history.append(
            pd.DataFrame(
                np.hstack((np.array([idx] * queries.shape[0])[:, None], queries, new_y)),
                columns=self.data_history.columns,
            ),
            ignore_index=True,
        )

    def _initialize_data_history(self):
        columns = ["step_index"]
        columns.extend([f"x{i}" for i in range(self.x_data.shape[1])])
        columns.extend(["y"])

        self.data_history = pd.DataFrame(
            np.hstack([np.zeros([self.x_data.shape[0], 1]), self.x_data, self.y_data]), columns=columns
        )

    @abc.abstractmethod
    def learn(self, n_steps: int):
        """
        Main Active learning loop - makes n_steps queries (calls the self.oracle object at each query) and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        raise NotImplementedError

    def save_experiment_summary(self):
        with pd.ExcelWriter(os.path.join(self.summary_path, self.summary_filename), mode="w") as writer:
            self.data_history.to_excel(writer, sheet_name="data")
            self.validation_metrics.to_excel(writer, sheet_name="result")


if __name__ == "__main__":
    pass
