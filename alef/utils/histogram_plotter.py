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

import matplotlib.pyplot as plt
import numpy as np
import os


class HistogramPlotter:
    def __init__(self, num_axes, v_axes=1):
        self.num_axes = num_axes
        self.num_v_axes = v_axes
        figsize = (4 * num_axes, 4 * v_axes)
        self.fig, self.axes = plt.subplots(v_axes, num_axes, figsize=figsize)

    def _add_histogram(
        self, *, values=None, bins=None, density=False, cumulative=False, color=None, label=None, ax_num=0, v_ax=0
    ):
        values = np.array(values)
        self.give_axes(ax_num, v_ax).hist(
            values, bins=bins, density=density, cumulative=cumulative, color=color, label=label
        )

    def add_histogram(self, values, bins, color, label, ax_num, v_ax=0):
        self._add_histogram(
            values=values,
            bins=bins,
            density=False,
            cumulative=False,
            color=color,
            label=label,
            ax_num=ax_num,
            v_ax=v_ax,
        )

    def add_histogram_density(self, values, bins, color, label, ax_num, v_ax=0):
        self._add_histogram(
            values=values, bins=bins, density=True, cumulative=False, color=color, label=label, ax_num=ax_num, v_ax=v_ax
        )

    def add_vline(self, value, color, ax_num, v_ax=0):
        if value != np.inf and value != -np.inf:
            self.give_axes(ax_num, v_ax).axvline(value, color=color, linestyle="--")

    def configure_axes(
        self,
        ax_num: int,
        v_ax: int = 0,
        ax_title: str = None,
        x_label: str = None,
        y_label: str = None,
        log_scale_y: bool = False,
        add_legend: bool = False,
    ):
        ax = self.give_axes(ax_num, v_ax)
        ax.set_xlabel(x_label)
        if log_scale_y:
            ax.set_yscale("log")
            ax.set_ylabel(y_label + " (log-scale)")
        else:
            ax.set_ylabel(y_label)
        ax.set_title(ax_title)
        if add_legend:
            ax.legend()

    def give_axes(self, ax_num, v_ax=0):
        if self.num_v_axes == 1:
            if self.num_axes > 1:
                return self.axes[ax_num]
            else:
                return self.axes
        else:
            if self.num_axes == 1:
                return self.axes[v_ax]
            else:
                return self.axes[v_ax, ax_num]

    def save_fig(self, file_path, file_name):
        plt.tight_layout()
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plotter = HistogramPlotter(1)
    plotter.add_histogram(
        [200, 100, 500, 700], [0.3, 0.5, 0.2, 0.15], [0.25, 0.45, 0.15, 0.1], [0.35, 0.55, 0.25, 0.2], "red", "HHK", 0
    )
    plotter.add_metrics_curve([200, 100, 500, 700], [0.4, 0.7, 0.15, 0.0], "blue", "RBF", 0)
    plotter.configure_axes(0, 0, "RMSE", "n_data", "RMSE", False, True)
    plotter.show()
