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


def f_s(x):
    xx = 1.9 * x - 1
    y = xx**3 - 1 / 2 * xx - 1
    return np.sin(10 * y) + np.sin(xx**2) - 0.5


def f(x):
    xx = 1.9 * x - 1
    y = xx**3 - 1 / 2 * xx - 1
    return np.sin(10 * y) + xx**2 / 3 - 0.5


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x = np.linspace(0, 1, 50)
    plt.plot(x, f_s(x), "-", color="y")
    plt.plot(x, f(x), "-", color="b")
    plt.show()
