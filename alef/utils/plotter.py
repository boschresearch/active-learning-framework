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
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class Plotter:
    def __init__(self, num_axes, v_axes=1, share_x=False, share_y=False):
        self.num_axes = num_axes
        self.num_v_axes = v_axes
        figsize = (5 * num_axes, 4 * v_axes)
        self.fig, self.axes = plt.subplots(v_axes, num_axes, sharex=share_x, sharey=share_y, figsize=figsize)

    def add_acquisition_score(self, x, score, color, ax_num, sort_x=True, v_ax=0):
        ax = self.give_axes(ax_num, v_ax).twinx()
        if sort_x:
            sorted_indexes = np.argsort(x)
            ax.plot(x[sorted_indexes], score[sorted_indexes], color=color, alpha=0.5)
        else:
            ax.plot(x, score, color=color, alpha=0.5)
        ax.set_ylabel("acquisition_score", color=color)
        ax.tick_params(axis="y", labelcolor=color)

    def add_gt_function(self, x, ground_truth, color, ax_num, sort_x=True, v_ax=0):
        if sort_x:
            sorted_indexes = np.argsort(x)
            self.give_axes(ax_num, v_ax).plot(x[sorted_indexes], ground_truth[sorted_indexes], color=color)
        else:
            self.give_axes(ax_num, v_ax).plot(x, ground_truth, color=color)

    def add_datapoints(self, x_data, y_data, color, ax_num, label=None, v_ax=0):
        self.give_axes(ax_num, v_ax).plot(x_data, y_data, "o", color=color, label=label)

    def add_datapoints_flex_style(self, x_data, y_data, color, mark, alpha, ax_num, sort_x=True, label=None, v_ax=0):
        if sort_x:
            sorted_indexes = np.argsort(x_data)
            self.give_axes(ax_num, v_ax).plot(
                x_data[sorted_indexes], y_data[sorted_indexes], mark, color=color, alpha=alpha, label=label
            )
        else:
            self.give_axes(ax_num, v_ax).plot(x_data, y_data, mark, color=color, alpha=alpha, label=label)

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

    def add_posterior_functions(self, x, predictions, ax_num, v_ax=0):
        num_predictions = predictions.shape[0]
        for i in range(0, num_predictions):
            self.give_axes(ax_num, v_ax).plot(x, predictions[i], color="r", linewidth="0.5")

    def add_predictive_dist(self, x, pred_mu, pred_sigma, ax_num, sort_x=True, v_ax=0):
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            pred_sigma = pred_sigma[sorted_index]
        axes = self.give_axes(ax_num, v_ax)
        axes.plot(x, pred_mu, color="g")
        axes.fill_between(x, pred_mu - pred_sigma, pred_mu + pred_sigma, alpha=0.8, color="b")
        axes.fill_between(x, pred_mu - 2 * pred_sigma, pred_mu + 2 * pred_sigma, alpha=0.3, color="b")

    def add_confidence_bound(self, x, pred_mu, bound_width, ax_num, sort_x=True, v_ax=0):
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            bound_width = bound_width[sorted_index]
        axes = self.give_axes(ax_num, v_ax)
        axes.plot(x, pred_mu, color="g")
        axes.fill_between(x, pred_mu - bound_width, pred_mu + bound_width, alpha=0.3, color="b")

    def add_multiple_confidence_bound(self, x, pred_mu, bound_width, ax_num, sort_x=True, v_ax=0):
        assert np.shape(pred_mu) == np.shape(bound_width)
        if len(np.shape(pred_mu)) == 1 or np.shape(pred_mu)[1] == 1:
            self.add_confidence_bound(x, np.squeeze(pred_mu), np.squeeze(bound_width), ax_num, sort_x, v_ax=v_ax)
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            bound_width = bound_width[sorted_index]
        axes = self.give_axes(ax_num, v_ax)
        for i in range(pred_mu.shape[1]):
            axes.plot(x, pred_mu[..., i], color=f"C{i}")
            axes.fill_between(
                x,
                pred_mu[..., i] - bound_width[..., i],
                pred_mu[..., i] + bound_width[..., i],
                alpha=0.3,
                color=f"C{i}",
            )

    def add_hyperparameters(self, hyperparameters, ax_num, v_ax=0, losses=None, labels=None, sort_loss=True):
        """
        param hyperparameters: numpy array with shape (num_hyperparameters, num_samples)
        param losses: value of loss objective for each hypp sample. shape: (num_samples,)
        """
        hyperparam_count = len(hyperparameters)
        if sort_loss and losses is not None:
            sorted_index = np.argsort(np.squeeze(losses))[::-1]
            hyperparameters = hyperparameters[:, sorted_index]
        if labels is None:
            labels = []
            for idx in range(hyperparam_count):
                labels.append("hyperparameter " + str(idx))
        else:
            assert len(losses) == hyperparam_count
        axes = self.give_axes(ax_num, v_ax)
        for idx, hyperparam in enumerate(hyperparameters):
            axes.plot(np.squeeze(hyperparam), label=labels[idx], marker=".", color=f"C{idx}")

    def add_hyperparameter_losses(self, losses, ax_num, sort_loss=True, v_ax=0):
        if sort_loss:
            sorted_index = np.argsort(losses)[::-1]
            losses = losses[sorted_index]
        self.give_axes(ax_num, v_ax).plot(losses, marker=".")

    def add_hline(self, y_value, color, ax_num, v_ax=0):
        if y_value != np.inf and y_value != -np.inf:
            self.give_axes(ax_num, v_ax).axhline(y_value, color=color, linewidth=0.8, linestyle="--")

    def add_multiple_hline(self, y_values, ax_num, v_ax=0):
        for i, y in enumerate(y_values):
            if y == np.inf or y == -np.inf:
                continue
            self.add_hline(y, f"C{i}", ax_num, v_ax=v_ax)

    def add_vline(self, x_value, color, ax_num, v_ax=0):
        if x_value != np.inf and x_value != -np.inf:
            self.give_axes(ax_num, v_ax).axvline(x_value, color=color, linestyle="--")

    def add_safety_region(self, safe_x, ax_num, v_ax=0):
        min_y = self.give_axes(ax_num, v_ax).get_ylim()[0]
        self.give_axes(ax_num, v_ax).plot(safe_x, np.repeat(min_y, safe_x.shape[0]), "_", linewidth=10.0, color="green")

    def add_query_region(self, query_x_grid, ax_num, v_ax=0):
        min_y = self.give_axes(ax_num, v_ax).get_ylim()[0]
        self.give_axes(ax_num, v_ax).plot(
            query_x_grid, np.repeat(min_y, query_x_grid.shape[0]), "_", linewidth=10.0, color="purple"
        )

    def save_fig(self, file_path, file_name):
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def show(self):
        plt.show()


class PlotterPlotly:
    def __init__(self, num_axes, share_x=False, share_y=False, width=800, height_per_axis=500) -> None:
        if share_y:
            share_y = "all"
        self.fig = make_subplots(num_axes, 1, shared_xaxes=share_x, shared_yaxes=share_y)
        self.width = width
        self.height = height_per_axis * num_axes
        self.show_legend(False)
        self.update_layout()

    def add_gt_function(self, x, ground_truth, color, ax_num, sort_x=True, line_opacity=1.0, name=""):
        x = np.squeeze(x)
        if name != "":
            showlegend = True
            print(showlegend)
        else:
            showlegend = False
        ground_truth = np.squeeze(ground_truth)
        if sort_x:
            sorted_indexes = np.argsort(x)
            x = x[sorted_indexes]
            ground_truth = ground_truth[sorted_indexes]

        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=ground_truth,
                line=go.scatter.Line(color=color),
                opacity=line_opacity,
                legendgroup=name,
                name=name,
                showlegend=showlegend,
            ),
            row=ax_num + 1,
            col=1,
        )

    def add_datapoints(self, x_data, y_data, color, ax_num):
        x_data = np.squeeze(x_data)
        y_data = np.squeeze(y_data)
        self.fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                marker=dict(size=8, color=color, line=dict(width=0.3, color="DarkSlateGrey")),
                mode="markers",
                showlegend=False,
            ),
            row=ax_num + 1,
            col=1,
        )

    def show_legend(self, show_it):
        self.fig.update_layout(showlegend=show_it)

    def add_predictive_dist(self, x, pred_mu, pred_sigma, ax_num, sort_x=True, opacity_scale=1.0):
        x = np.squeeze(x)
        pred_mu = np.squeeze(pred_mu)
        pred_sigma = np.squeeze(pred_sigma)
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            pred_sigma = pred_sigma[sorted_index]
        upper_sigma = pred_mu + pred_sigma
        upper_two_sigma = pred_mu + 2 * pred_sigma
        lower_sigma = pred_mu - pred_sigma
        lower_two_sigma = pred_mu - 2 * pred_sigma
        color = "blue"
        color = "51, 59, 255"
        self.fig.add_trace(
            go.Scatter(
                x=x, y=upper_two_sigma, line=dict(width=0), mode="lines", opacity=opacity_scale, showlegend=False
            ),
            row=ax_num + 1,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=upper_sigma,
                line=dict(width=0),
                fillcolor=f"rgba({color},{0.3 * opacity_scale})",
                mode="lines",
                fill="tonexty",
                showlegend=False,
            ),
            row=ax_num + 1,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=pred_mu,
                line=go.scatter.Line(color="blue"),
                fillcolor=f"rgba({color},{0.6 * opacity_scale})",
                mode="lines",
                fill="tonexty",
                showlegend=False,
            ),
            row=ax_num + 1,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=lower_sigma,
                line=dict(width=0),
                fillcolor=f"rgba({color},{0.6 * opacity_scale})",
                mode="lines",
                fill="tonexty",
                showlegend=False,
            ),
            row=ax_num + 1,
            col=1,
        )
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=lower_two_sigma,
                line=dict(width=0),
                fillcolor=f"rgba({color},{0.3 * opacity_scale})",
                mode="lines",
                fill="tonexty",
                showlegend=False,
            ),
            row=ax_num + 1,
            col=1,
        )

    def set_x_axes(self, x_min, x_max):
        self.fig.update_layout(xaxis_range=[x_min, x_max])

    def save_fig(self, file_path, file_name):
        # self.update_layout()
        self.fig.write_image(os.path.join(file_path, file_name))

    def show(self):
        # self.update_layout()
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": "plot",
                "height": self.height,
                "width": self.width,
                "scale": 6,
            }
        }
        self.fig.show(config=config)

    def update_layout(self):
        self.fig.update_layout(autosize=False, width=self.width, height=self.height)
        # self.fig.update_layout(showlegend=False)
        self.fig.update_xaxes(title="x", automargin=False, nticks=10)
        self.fig.update_yaxes(title="y", automargin=False)

        self.fig.update_layout(font_size=15)


if __name__ == "__main__":
    plotter = PlotterPlotly(1)
    x = np.random.uniform(0, 1, (20, 1))
    y = np.random.uniform(0, 1, (20, 1))
    x2 = np.random.uniform(0, 1, (10, 1))
    y2 = np.random.uniform(0, 1, (10, 1))
    pred_sigma = np.zeros((10)) + 0.2

    plotter.add_datapoints(np.squeeze(x), np.squeeze(y), "green", 0)
    plotter.add_datapoints(np.squeeze(x2), np.squeeze(y2), "red", 0)

    plotter.add_predictive_dist(np.squeeze(x2), np.squeeze(y2), pred_sigma, 0, True)
    plotter.show()
