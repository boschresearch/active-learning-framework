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


class MetricCurvePlotter:
    def __init__(self, num_axes):
        self.num_axes = num_axes
        figsize = (8 * num_axes, 4)
        self.fig, self.axes = plt.subplots(1, num_axes, figsize=figsize)

    def add_metrics_curve(self, x_values, metrics_value, color, label, ax_num, sort_x=True):
        x_values = np.array(x_values)
        metrics_value = np.array(metrics_value)
        if sort_x:
            sorted_indexes = np.argsort(x_values)
            self.give_axes(ax_num).plot(
                x_values[sorted_indexes], metrics_value[sorted_indexes], color=color, label=label
            )
        else:
            self.give_axes(ax_num).plot(x_values, metrics_value, color=color, label=label)

    def add_metrics_curve_with_errors(
        self, x_values, metrics_value, lower_error_bar, upper_error_bar, color, label, ax_num, sort_x=True
    ):
        x_values = np.array(x_values)
        metrics_value = np.array(metrics_value)
        lower_error_bar = metrics_value - np.array(lower_error_bar)
        upper_error_bar = np.array(upper_error_bar) - metrics_value
        if sort_x:
            sorted_indexes = np.argsort(x_values)
            self.give_axes(ax_num).plot(
                x_values[sorted_indexes], metrics_value[sorted_indexes], color=color, label=label
            )
            self.give_axes(ax_num).errorbar(
                x_values[sorted_indexes],
                metrics_value[sorted_indexes],
                yerr=[lower_error_bar[sorted_indexes], upper_error_bar[sorted_indexes]],
                marker="s",
                color=color,
                capsize=4,
            )
        else:
            self.give_axes(ax_num).plot(x_values, metrics_value, color=color, label=label)
            self.give_axes(ax_num).errorbar(
                x_values, metrics_value, yerr=[lower_error_bar, upper_error_bar], marker="s", color=color, capsize=4
            )

    def configure_axes(
        self,
        ax_num: int,
        ax_title: str,
        x_label: str,
        y_label: str,
        log_scale_y: bool,
        add_legend: bool,
        x_lim=None,
        y_lim=None,
    ):
        ax = self.give_axes(ax_num)
        ax.set_xlabel(x_label)
        if log_scale_y:
            ax.set_yscale("log")
            ax.set_ylabel(y_label + " (log-scale)")
        else:
            ax.set_ylabel(y_label)
        ax.set_title(ax_title)
        if add_legend:
            ax.legend()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    def give_axes(self, ax_num):
        if self.num_axes > 1:
            return self.axes[ax_num]
        else:
            return self.axes

    def save_fig(self, file_path, file_name):
        plt.tight_layout()
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plotter = MetricCurvePlotter(1)
    plotter.add_metrics_curve_with_errors(
        [200, 100, 500, 700], [0.3, 0.5, 0.2, 0.15], [0.25, 0.45, 0.15, 0.1], [0.35, 0.55, 0.25, 0.2], "red", "HHK", 0
    )
    plotter.add_metrics_curve([200, 100, 500, 700], [0.4, 0.7, 0.15, 0.0], "blue", "RBF", 0)
    plotter.configure_axes(0, "RMSE", "n_data", "RMSE", False, True)
    plotter.show()
