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
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


class MnistSVM2d:
    def __init__(self) -> None:
        self.transform = True
        if self.transform:
            self.__a = -1.0
            self.__b = 0.0
        else:
            self.__a = 0.000001
            self.__b = 1.0
        self.load_data()

    def load_data(self):
        mnist_data = datasets.load_digits()
        images = mnist_data.images.reshape((len(mnist_data.images), -1))
        labels = mnist_data.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=0.2, shuffle=False
        )

    def query(self, x: np.array) -> np.float:
        if self.transform:
            gamma = np.exp(10 * x[0])
            C = (x[1] + 1.000001) * 2.0
        else:
            gamma = x[0]
            C = x[1] * 2.0
        svm_classifier = svm.SVC(gamma=gamma, C=C)
        svm_classifier.fit(self.x_train, self.y_train)
        pred_test = svm_classifier.predict(self.x_test)
        accuracy = metrics.accuracy_score(self.y_test, pred_test)
        return accuracy

    def get_random_data(self, n: int, noisy: bool = False) -> Tuple[np.array, np.array]:
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.get_dimension()))
        function_values = []
        for x in X:
            function_value = self.query(x)
            function_values.append(function_value)
        print("-Dataset of length " + str(n) + " generated")
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return 2


if __name__ == "__main__":
    mnist_svm = MnistSVM2d()
    mnist_svm.load_data()
    print(mnist_svm.get_random_data(6))
