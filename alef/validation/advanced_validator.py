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
from typing import List, Optional
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import alef.validation.plot_dicts as plot_dicts
import matplotlib
import pandas as pd
import seaborn as sns
from scipy import stats


class PlotType(Enum):
    MEDIAN_INTERVAL = 0
    MEAN = 1


def set_font_sizes(font_size, only_axis=True):
    if only_axis:
        matplotlib.rc("xtick", labelsize=font_size)
        matplotlib.rc("ytick", labelsize=font_size)
    else:
        font = {"family": "normal", "size": font_size}

        matplotlib.rc("font", **font)


class SingleRun:
    def __init__(self, run_name: str, seed_value: int) -> None:
        self.run_name = run_name
        self.seed_value = seed_value
        self.main_metric_array = None
        self.iteration_time_array = None
        self.limit_iterations = False
        self.iteration_limit = 50

    def set_main_metric(self, main_metric_array):
        self.main_metric_array = main_metric_array

    def get_main_metric_array(self):
        return self.main_metric_array

    def set_iteration_time(self, iteration_time_array):
        if self.limit_iterations:
            self.iteration_time_array = iteration_time_array[: self.iteration_limit]
        else:
            self.iteration_time_array = iteration_time_array

    def get_iteration_time(self):
        return self.iteration_time_array

    def get_metric_array_over_time_array(self, interval=1.0, bound_on_time=3000):
        cummulative_time_list = []
        elapsed_time = 0.0
        cummulative_time_list.append(elapsed_time)
        for iteration_time in self.iteration_time_array:
            elapsed_time += iteration_time
            cummulative_time_list.append(elapsed_time)
        num_time_steps = int(elapsed_time / interval) + 1
        time_points = [step * interval for step in range(0, num_time_steps)]
        metrics_over_time_points = []
        time_indexes = []
        for time_point in time_points:
            if time_point > bound_on_time:
                print("Name {} Seed {} Index last time point: {}".format(self.run_name, self.seed_value, intex_t_1))  # noqa: F821
                break
            time_indexes.append(time_point)
            index_t = 0
            intex_t_1 = 0
            time_diff_to_t = 0.0
            for index, cummalitive_time in enumerate(cummulative_time_list):
                if time_point < cummulative_time_list[index + 1]:
                    index_t = index
                    intex_t_1 = index + 1
                    time_diff_to_t = time_point - cummalitive_time
                    time_diff_cummulative = cummulative_time_list[index + 1] - cummalitive_time
                    break
            metric_over_time_point = self.main_metric_array[index_t] + (
                self.main_metric_array[intex_t_1] - self.main_metric_array[index_t]
            ) * (float(time_diff_to_t) / float(time_diff_cummulative))
            metrics_over_time_points.append(metric_over_time_point)
        last_index = intex_t_1
        return np.array(metrics_over_time_points), np.array(time_indexes), last_index


class RunCollection:
    def __init__(self, run_name: str):
        self.run_dict = {}
        self.run_name = run_name

    def set_run_dict(self, run_dict):
        self.run_dict = run_dict

    def get_last_index_dict(self, interval, time_bound):
        last_index_dict = {}
        for run_id in self.run_dict:
            metrics_over_time, time_array, last_index = self.run_dict[run_id].get_metric_array_over_time_array(
                interval, time_bound
            )
            seed_value = self.run_dict[run_id].seed_value
            last_index_dict[seed_value] = last_index
        return last_index_dict

    def get_run_length(self):
        metric_matrix = self.get_metric_matrix()
        return metric_matrix.shape[1]

    def get_metric_matrix(self):
        metrics_list = []
        for run_id in self.run_dict:
            metrics_list.append(self.run_dict[run_id].get_main_metric_array())
        len_list = [len(metrics) for metrics in metrics_list]
        print(self.run_name)
        print(len_list)
        min_len = min(len_list)
        max_len = max(len_list)
        if max_len > min_len:
            new_metrics_list = [metrics[:min_len] for metrics in metrics_list]
            return np.array(new_metrics_list)
        return np.array(metrics_list)

    def get_metric_matrix_over_time(self, interval, time_bound):
        metrics_list = []
        for run_id in self.run_dict:
            metrics_over_time, time_array, _ = self.run_dict[run_id].get_metric_array_over_time_array(
                interval, time_bound
            )
            metrics_list.append(metrics_over_time)
        len_list = [len(metrics) for metrics in metrics_list]
        min_len = min(len_list)
        max_len = max(len_list)
        if max_len > min_len:
            new_metrics_list = [metrics[:min_len] for metrics in metrics_list]
            return np.array(new_metrics_list), time_array[:min_len]
        return np.array(metrics_list), time_array

    def get_mean_std_array(self):
        metric_matrix = self.get_metric_matrix()
        mean_array = np.mean(metric_matrix, axis=0)
        std_array = np.std(metric_matrix, axis=0)
        return mean_array, std_array

    def get_quartile_arrays(self):
        metric_matrix = self.get_metric_matrix()
        median_array = np.median(metric_matrix, axis=0)
        lower_quartile_array = np.quantile(metric_matrix, 0.25, axis=0)
        upper_quartile_array = np.quantile(metric_matrix, 0.75, axis=0)
        return lower_quartile_array, median_array, upper_quartile_array

    def get_mean_std_array_over_time(self, interval, time_bound):
        metric_matrix, time_array = self.get_metric_matrix_over_time(interval, time_bound)
        mean_array = np.mean(metric_matrix, axis=0)
        std_array = np.std(metric_matrix, axis=0)
        return mean_array, std_array, time_array

    def get_quartile_arrays_over_time(self, interval, time_bound):
        metric_matrix, time_array = self.get_metric_matrix_over_time(interval, time_bound)
        median_array = np.median(metric_matrix, axis=0)
        lower_quartile_array = np.quantile(metric_matrix, 0.25, axis=0)
        upper_quartile_array = np.quantile(metric_matrix, 0.75, axis=0)
        return lower_quartile_array, median_array, upper_quartile_array, time_array


class IndexSettings:
    def __init__(
        self,
        use_subindexes=False,
        subindex_factor=10,
        bound_indexes=False,
        index_bound=300,
        add_first_index=False,
        scale_index=False,
        index_scale=10,
    ) -> None:
        self.use_subindexes = use_subindexes
        self.bound_indexes = bound_indexes
        self.index_bound = index_bound
        self.add_first_index = add_first_index
        self.subindex_factor = subindex_factor
        self.scale_index = scale_index
        self.index_scale = index_scale


class TimeSettings:
    def __init__(self, interval, time_bound):
        self.interval = interval
        self.time_bound = time_bound


class AdvancedValidator:
    def __init__(self, only_include_run_names: bool = False, run_names: Optional[List] = None):
        self.only_include_run_names = only_include_run_names
        self.run_names_included = run_names
        self.run_dict = {}
        self.run_collections = {}
        self.alpha = 0.05
        self.color_dict = plot_dicts.color_dict
        self.name_dict = plot_dicts.name_dict
        # self.color_list=['navy','blue','dodgerblue','aqua','green','red','peru','olive','aqua','pink','lime']
        self.color_list = ["green", "navy", "red", "orange", "orange", "black"]

    def collect_from_folder(self, folder, filter_for_metric_name=False, metric_name_filter=None):
        file_names = [file_name for file_name in os.listdir(folder) if file_name.endswith(".txt")]
        for file_name in file_names:
            run_name = file_name.split("_")[0]
            run_id = int(file_name.split("_")[1])
            metric_name = file_name.split("_")[2]
            if filter_for_metric_name:
                if metric_name == metric_name_filter:
                    metric_array = np.loadtxt(os.path.join(folder, file_name))
                    self.add_run(metric_array, run_name, run_id)
            else:
                metric_array = np.loadtxt(os.path.join(folder, file_name))
                self.add_run(metric_array, run_name, run_id)

    def collect_metrics_from_subfolders(self, folder, metrics_file_name, index=0, skiprows=0):
        folder_names = [folder_name for folder_name in os.listdir(folder)]
        for sub_folder_name in folder_names:
            run_name = sub_folder_name.split("_")[0]
            run_id = int(sub_folder_name.split("_")[1])
            file_path = os.path.join(folder, sub_folder_name, metrics_file_name)
            if os.path.isfile(file_path):
                metric_array = np.loadtxt(file_path, skiprows=skiprows, delimiter=",")
                if len(metric_array.shape) == 2:
                    metric_array = metric_array[:, index]
                self.add_run(metric_array, run_name, run_id)

    def collect_metrics_from_file_from_subfolders_for_run(
        self,
        folder,
        run_name,
        metrics_file_name,
        skiprows=0,
        delimiter=" ",
        index_list=[0, 1, 2],
        name_list=["alpha1", "alpha2", "alpha3"],
        normalize=False,
    ):
        folder_names = [folder_name for folder_name in os.listdir(folder) if folder_name.split("_")[0] == run_name]
        for sub_folder_name in folder_names:
            run_id = int(sub_folder_name.split("_")[1])
            file_path = os.path.join(folder, sub_folder_name, metrics_file_name)
            if os.path.isfile(file_path):
                metric_array = np.loadtxt(file_path, skiprows=skiprows, delimiter=delimiter)[:, index_list]
                if normalize:
                    metric_array = metric_array / np.expand_dims(np.sum(metric_array, axis=1), axis=1)

                if len(metric_array.shape) == 2:
                    for index in index_list:
                        single_metric_array = metric_array[:, index]
                        self.add_run(single_metric_array, name_list[index], run_id)
                else:
                    self.add_run(metric_array, name_list[0], run_id)

    def collect_iteration_time_from_subfolders(self, folder, time_file_name, cummulate_time=False, cummulation_n=10):
        folder_names = [folder_name for folder_name in os.listdir(folder)]
        for sub_folder_name in folder_names:
            run_name = sub_folder_name.split("_")[0]
            print(run_name)
            run_id = int(sub_folder_name.split("_")[1])
            print(run_id)
            file_path = os.path.join(folder, sub_folder_name, time_file_name)
            if os.path.isfile(file_path):
                time_array = np.loadtxt(file_path)
                if cummulate_time:
                    time_array = self.cummulate_time_array(time_array, cummulation_n)
                if run_name in self.run_dict:
                    if run_id in self.run_dict[run_name]:
                        self.run_dict[run_name][run_id].set_iteration_time(time_array)

    def cummulate_time_array(self, time_array, n_cummulate):
        ##Cummulates iteration time array in the sense that it decreases the resolution -e.g. the iteration times [1.0,2.0,1.0,0.5] are summarizes to [3.0,1.5] for n_cummulate=2
        cummulated_iteration_time_list = []
        cumm_iteration_time = 0.0
        for i, iteration_time in enumerate(time_array):
            cumm_iteration_time += iteration_time
            if (i + 1) % n_cummulate == 0:
                cummulated_iteration_time_list.append(cumm_iteration_time)
                cumm_iteration_time = 0.0

        return np.array(cummulated_iteration_time_list)

    def add_run(self, metric_array, run_name, run_id):
        if self.only_include_run_names and (not run_name in self.run_names_included):
            return
        if np.isnan(metric_array).any():
            return
        new_run = SingleRun(run_name, run_id)
        new_run.set_main_metric(metric_array)
        if not run_name in self.run_dict:
            self.run_dict[run_name] = {}
            self.run_dict[run_name][run_id] = new_run
        else:
            assert not (run_id in self.run_dict[run_name])
            self.run_dict[run_name][run_id] = new_run

    def prune_runs(self, only_with_iteration_time=False, use_only_first_ids=False, n_first=30):
        """
        Method to use only a subset of run ids - its main job is to prune the runs to the ids that are present in all run_names
        it can also be used to only use the ids where also an iteration time array is present or to use only the first n_first ids
        """
        print("-Prune dict")
        counter = 0
        if only_with_iteration_time:
            new_run_dict = {}
            for run_name in self.run_dict:
                run_name_dict = {}
                for run_id in self.run_dict[run_name]:
                    if self.run_dict[run_name][run_id].iteration_time_array is not None:
                        run_name_dict[run_id] = self.run_dict[run_name][run_id]
                    else:
                        print(run_name)
                        print(run_id)
                        assert False
                if len(run_name_dict) > 0:
                    new_run_dict[run_name] = run_name_dict
            self.run_dict = new_run_dict
        # filter out ids that are not present in all runs
        for run_name in self.run_dict:
            if counter == 0:
                ids = set(self.run_dict[run_name].keys())
            else:
                ids = ids.intersection(set(self.run_dict[run_name].keys()))
            counter += 1
        print("IDS:")
        print(ids)
        print(len(ids))
        if use_only_first_ids:
            id_list = list(ids)
            id_list.sort()
            ids = set(id_list[:n_first])
            print("First ids")
            print(ids)

        new_run_dict = {}
        for run_name in self.run_dict:
            new_run_dict[run_name] = {}
            for run_id in self.run_dict[run_name]:
                if run_id in ids:
                    new_run_dict[run_name][run_id] = self.run_dict[run_name][run_id]
        self.run_dict = new_run_dict
        self.show_run_dict()

    def create_run_collections(self):
        for run_name in self.run_dict:
            new_collection = RunCollection(run_name)
            new_collection.set_run_dict(self.run_dict[run_name])
            self.run_collections[run_name] = new_collection

    def show_run_dict(self):
        for run_name in self.run_dict:
            print(run_name)
            print(self.run_dict[run_name].keys())
            print("Length: " + str(len(self.run_dict[run_name].keys())))

    def get_indexes(self, run_name, index_settings: IndexSettings):
        run_length = self.run_collections[run_name].get_run_length()
        indexes = np.array(list(range(0, run_length)))
        if index_settings.use_subindexes:
            index_mask = [(index + 1) % index_settings.subindex_factor == 0 for index in indexes]
            if index_settings.add_first_index:
                index_mask[0] = True
            indexes = indexes[index_mask]
        if index_settings.bound_indexes:
            indexes = indexes[indexes <= index_settings.index_bound]
        return indexes

    def create_plot(
        self,
        plot_type,
        metric_name,
        log_scale: bool,
        index_settings: IndexSettings,
        use_time_as_index: bool,
        time_settings: TimeSettings,
        x_label="Iteration",
        min_max_y=None,
        return_fig=False,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        color_counter = 0
        if self.only_include_run_names:
            run_names = self.run_names_included
        else:
            run_names = list(self.run_collections.keys())

        for run_name in run_names:
            if run_name in self.color_dict:
                color = self.color_dict[run_name]
            else:
                color = self.color_list[color_counter]
                color_counter += 1

            if run_name in self.name_dict:
                label = self.name_dict[run_name]
            else:
                label = run_name

            indexes = self.get_indexes(run_name, index_settings)

            if plot_type == PlotType.MEDIAN_INTERVAL:
                if use_time_as_index:
                    lower_quartile_array, median_array, upper_quartile_array, time_array = self.run_collections[
                        run_name
                    ].get_quartile_arrays_over_time(time_settings.interval, time_settings.time_bound)
                    mean_array, std_array, time_array = self.run_collections[run_name].get_mean_std_array_over_time(
                        time_settings.interval, time_settings.time_bound
                    )
                    errors = np.array([median_array - lower_quartile_array, upper_quartile_array - median_array])
                    ax.plot(time_array, median_array, color=color, label=label)
                    ax.errorbar(time_array, median_array, yerr=errors, marker="s", color=color, capsize=4)
                    print(run_name)
                    print(median_array)
                    print("Lenght: " + str(len(median_array)))
                    print("Last step:")
                    print("Median:")
                    print(median_array[-1])
                    print("Mean:")
                    print(mean_array[-1])
                    print("Std:")
                    print(std_array[-1])
                else:
                    lower_quartile_array, median_array, upper_quartile_array = self.run_collections[
                        run_name
                    ].get_quartile_arrays()
                    errors = np.array(
                        [
                            median_array[indexes] - lower_quartile_array[indexes],
                            upper_quartile_array[indexes] - median_array[indexes],
                        ]
                    )
                    if index_settings.scale_index:
                        index_values = indexes * index_settings.index_scale
                    else:
                        index_values = indexes
                    ax.plot(index_values, median_array[indexes], color=color, label=label)
                    ax.errorbar(index_values, median_array[indexes], yerr=errors, marker="s", color=color, capsize=4)
            elif plot_type == PlotType.MEAN:
                if use_time_as_index:
                    raise NotImplementedError
                else:
                    mean_array, std_array = self.run_collections[run_name].get_mean_std_array()
                    errors = np.array([std_array[indexes], std_array[indexes]])
                    if index_settings.scale_index:
                        index_values = indexes * index_settings.index_scale
                    else:
                        index_values = indexes
                    ax.plot(index_values, mean_array[indexes], color=color, label=label)
                    # ax.errorbar(index_values, mean_array[indexes], yerr=errors, marker="s", color=color, capsize=4)

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel(metric_name + " (log-scale)")
        else:
            ax.set_ylabel(metric_name)
        if min_max_y is not None:
            min_y, max_y = min_max_y
            ax.set_ylim(min_y, max_y)
        ax.set_xlabel(x_label)
        ax.set_title(metric_name)
        if return_fig:
            return fig, ax
        ax.legend()
        plt.tight_layout()
        plt.show()

    def create_box_plots_at_step(
        self, metric_name: str, iteration: int, use_time: bool, time_settings: TimeSettings, time_bound: int
    ):
        color_counter = 0
        if self.only_include_run_names:
            run_names = self.run_names_included
        else:
            run_names = list(self.run_collections.keys())

        metrics_dict = {}

        for run_name in run_names:
            if run_name in self.color_dict:
                self.color_dict[run_name]
            else:
                self.color_list[color_counter]
                color_counter += 1

            if run_name in self.name_dict:
                label = self.name_dict[run_name].split(" ")[0]

            else:
                label = run_name

            if use_time:
                metrics_array_at_time_bound = self.run_collections[run_name].get_metric_matrix_over_time(
                    time_settings.interval, time_bound
                )[0][:, -1]
                metrics_dict[label] = metrics_array_at_time_bound
            else:
                metrics_array_at_iteration = self.run_collections[run_name].get_metric_matrix()[:, iteration]
                metrics_dict[label] = metrics_array_at_iteration

        create_boxplot(metric_name, metrics_dict)


def create_boxplot(metric_name, metrics_dict):
    print(metrics_dict)
    metrics_df = pd.DataFrame.from_dict(metrics_dict)

    print("Means:")
    print(metrics_df.mean(axis=0))
    print("Stds:")
    print(metrics_df.std(axis=0))
    print("Medians:")
    print(metrics_df.median(axis=0))
    print("Min mean:")
    minvalueIndexLabel = metrics_df.mean(axis=0).idxmin(axis=0)
    print(minvalueIndexLabel)

    for column in metrics_df.columns:
        if not column == minvalueIndexLabel:
            print(column)
            print(stats.ttest_rel(metrics_df[column], metrics_df[minvalueIndexLabel]))
    print("Means latex")
    print(metrics_df.mean(axis=0).to_frame().transpose().to_latex(escape=False))
    print("Std latex")
    print(metrics_df.std(axis=0).to_frame().transpose().to_latex(escape=False))
    # sns.set_theme(style="whitegrid")
    ax = sns.boxplot(data=metrics_df)
    ax.set_ylabel(metric_name)
    # ax.set_ylim(0.025, 0.10)
    plt.tight_layout()
    plt.show()
