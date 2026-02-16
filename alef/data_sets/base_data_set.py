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
from typing import Tuple
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    @abstractmethod
    def load_data_set(self):
        """
        loads dataset (probably from some file - implementation dependent)
        """
        raise NotImplementedError

    @abstractmethod
    def get_complete_dataset(self, **kwargs) -> Tuple[np.array, np.array]:
        """
        Retrieves the complete dataset (of size n with input dimensions d and output dimensions m) as numpy arrays

        Returns
        np.array - x (input values) with shape (n,d)
        np.array - y (output values) with shape (n,m)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int, **kwargs) -> Tuple[np.array, np.array]:
        """
        retrieves sample of size n from dataset (without replacement)

        Returns
        np.array - x (input values) with shape (n,d)
        np.array - y (output values) with shape (n,m)
        """
        raise NotImplementedError

    @abstractmethod
    def sample_train_test(self, use_absolute: bool, n_train: int, n_test: int, fraction_train: float):
        """
        retrieves train and test data (mutually exclusive samples) either in absoulte numbers or as fraction from the complete dataset

        Arguments:
            use_absolute - bool specifying if absolute numbers of training and test data should be used or fraction of complete dataset
            n_train - int specifying how many training datapoints should be sampled
            n_test - int scpecifying how many test datapoints should be sampled
            fraction_train - fraction of complete dataset that is used as training data

        Returns
        np.array - x train data with shape (n_train,d)
        np.array - y train data with shape (n_train,m)
        np.array - x test data with shape (n_test,d)
        np.array - y test data with shape (n_test,m)
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self):
        """
        method to get name of dataset
        """
        raise NotImplementedError


class StandardDataSet(BaseDataset):
    def __init__(self):
        self.x: np.array
        self.y: np.array
        self.length: int
        self.name: str

    def get_complete_dataset(self):
        return self.x, self.y

    def sample(self, n):
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        y_sample = self.y[indexes]
        return x_sample, y_sample

    def sample_train_test(self, use_absolute: bool, n_train: int, n_test: int, fraction_train: float):
        r"""
        @ Mathias
        I propose to modify this method.
        Since your n_train & n_test and fraction_train won't be used at the same time.
        My code doesn't use this method so it's up to you.

        I propose 2 ways:
        1. change the input arguments to force the last 3 arguments kwargs,
            and set default values to them.
        (
            self,
            use_absolute: bool,
            *
            n_train: int=None,
            n_test: int=None,
            fraction_train: float=0
        )

        2. write 2 different methods, one takes n_train & n_test while the other takes fraction_train.
            could even write another hidden method called by these 2 methods,
            which does line 126-134, to reduce some redundancy.

        """
        if use_absolute:
            assert n_train < self.length
            n = n_train + n_test
            if n > self.length:
                n = self.length
                print("Test + Train set exceeds number of datapoints - use n-n_train test points")
        else:
            n = self.length
            n_train = int(fraction_train * n)
            n_test = n - n_train
        indexes = np.random.choice(self.length, n, replace=False)
        train_indexes = indexes[:n_train]
        assert len(train_indexes) == n_train
        test_indexes = indexes[n_train:]
        if use_absolute and n_train + n_test <= self.length:
            assert len(test_indexes) == n_test
        x_train = self.x[train_indexes]
        y_train = self.y[train_indexes]
        x_test = self.x[test_indexes]
        y_test = self.y[test_indexes]
        return x_train, y_train, x_test, y_test

    def get_name(self):
        return self.name


class DatasetWrapper(StandardDataSet):
    def __init__(self, x_data, y_data, name):
        super().__init__()
        self.x = x_data
        self.y = y_data
        self.length = len(self.x)
        self.name = name

    def load_data_set(self):
        pass
