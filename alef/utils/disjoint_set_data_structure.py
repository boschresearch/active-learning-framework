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

"""
This data structure is widely used.
The code is adapted from https://en.wikipedia.org/wiki/Disjoint-set_data_structure

Or check the followings:

Robert E. Tarjan and Jan van Leeuwen. 1984.
Worst-case Analysis of Set Union Algorithms.
J. ACM 31, 2 (April 1984), 245–281. https://doi.org/10.1145/62.2160

Harold N. Gabow, Robert Endre Tarjan, 1985.
A linear-time algorithm for a special case of disjoint set union.
Journal of Computer and System Sciences 30, 2, 209-221. https://doi.org/10.1016/0022-0000(85)90014-5
"""

import numpy as np
import pandas as pd


class DisjointSet:
    def __init__(self, X: list):
        self.__df = pd.DataFrame({x: [1, x] for x in X}, index=["size", "parent"])

    def _report_element_nonexist(self, x, make_set: bool = False):
        if make_set:
            self.make_new_set(x)
            return
        try:
            self.__df[x]
        except:
            raise ValueError("element does not exist, use 'make_new_set(*)' to add it")

    def get_sets(self):
        return np.unique([self.find(x) for x in self.__df.loc["parent"].unique()])

    def count_sets(self):
        return len(self.get_sets())

    def parent(self, x):
        self._report_element_nonexist(x, make_set=False)
        return self.__df.loc["parent", x]

    def set_parent(self, x, value):
        self._report_element_nonexist(x, make_set=False)
        self.__df.loc["parent", x] = value

    def size(self, x):
        self._report_element_nonexist(x, make_set=False)
        return self.__df.loc["size", x]

    def set_size(self, x, value):
        self._report_element_nonexist(x, make_set=False)
        self.__df.loc["size", x] = value

    def make_new_set(self, x):
        try:
            self.__df[x]
        except:
            self.__df[x] = [1, x]
        return

    def find(self, x):
        if self.parent(x) != x:
            self.set_parent(x, self.find(self.parent(x)))
        return self.parent(x)

    def union(self, x, y, make_set: bool = False):
        self._report_element_nonexist(x, make_set=make_set)
        self._report_element_nonexist(y, make_set=make_set)

        x = self.find(x)
        y = self.find(y)

        if x == y:
            return
        if self.size(x) < self.size(y):
            x, y = y, x
        self.set_parent(y, x)
        self.set_size(x, self.size(x) + self.size(y))
        return


if __name__ == "__main__":
    # Driver code

    obj = DisjointSet([0, 2, 3, 7, 8, 10, 12])
    obj.union(2, 0)
    obj.union(3, 1, make_set=True)
    obj.union(8, 12)
    obj.union(1, 12)

    print(obj.get_sets())

    for i in range(13):
        try:
            print(f"size: {obj.size(obj.find(i))}  repr: {obj.find(i)}")
        except:
            print(f"element {i} does not exist")
