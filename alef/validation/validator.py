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

from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import wilcoxon
import os
import matplotlib.patches as mpatches

import alef.validation.plot_dicts as plot_dicts


class PlotType(Enum):
    CONFIDENCE_BAND = 1
    CONFIDENCE_INTERVALLS = 2
    MEAN = 3
    BOX_PLOTS = 4
    MEDIAN_INTERVAL = 5


class Validator:
    def __init__(self):
        self.metrics_dict = {}
        self.means_dict = {}
        self.median_dict = {}
        self.upper_quantile_dict = {}
        self.lower_quantile_dict = {}
        self.sigma_dict = {}
        self.upper_ki_dict = {}
        self.lower_ki_dict = {}
        self.complete_metric_array_dict = {}
        # self.median_dict={}
        # self.upper_quartile_dict={}
        # self.lower_quartile_dict={}
        # self.lower_whisker_dict={}
        # self.upper_whisker_dict={}
        self.alpha = 0.05
        self.color_dict = plot_dicts.color_dict
        self.name_dict = plot_dicts.name_dict
        # self.color_list=['navy','blue','dodgerblue','aqua','green','red','peru','olive','aqua','pink','lime']
        self.color_list = ["green", "black", "red", "orange", "orange", "black"]
        self.include_test = False
        self.test_results = {}

    def collect_from_folder(self, folder, filter_for_metric_name=False, metric_name_filter=None):
        file_names = [file_name for file_name in os.listdir(folder) if file_name.endswith(".txt")]
        for file_name in file_names:
            query_name = file_name.split("_")[0]
            run_id = int(file_name.split("_")[1])
            metric_name = file_name.split("_")[2]
            if filter_for_metric_name:
                if metric_name == metric_name_filter:
                    metric_array = np.loadtxt(os.path.join(folder, file_name))
                    self.add_metric_array(metric_array, query_name, run_id)
            else:
                metric_array = np.loadtxt(os.path.join(folder, file_name))
                self.add_metric_array(metric_array, query_name, run_id)

    def collect_from_subfolders(
        self, folder, file_name, metric_name, index=0, skiprows=0, use_query_list=False, query_name_list=[]
    ):
        folder_names = [folder_name for folder_name in os.listdir(folder)]
        for sub_folder_name in folder_names:
            query_name = sub_folder_name.split("_")[0]
            run_id = sub_folder_name.split("_")[1]
            file_path = os.path.join(folder, sub_folder_name, file_name)
            if not use_query_list or query_name in query_name_list:
                if os.path.isfile(file_path):
                    metric_array = np.loadtxt(file_path, skiprows=skiprows, delimiter=",")
                    if len(metric_array.shape) == 2:
                        metric_array = metric_array[:, index]
                    self.add_metric_array(metric_array, query_name, run_id)

    def add_metric_array(self, metric_array, query_name, run_id):
        if not query_name in self.metrics_dict:
            self.metrics_dict[query_name] = {}
            self.metrics_dict[query_name][run_id] = metric_array
        else:
            print(run_id)
            print(query_name)
            assert not (run_id in self.metrics_dict[query_name])
            self.metrics_dict[query_name][run_id] = metric_array

    def dict_to_list(self, dictionary):
        new_list = []
        for key in dictionary:
            element = dictionary[key]
            new_list.append(element)
        return new_list

    def run_wilcoxon_test(self, query_name_1, query_name_2, exclude_initial=True):
        assert query_name_1 in self.metrics_dict
        assert query_name_2 in self.metrics_dict
        self.include_test = True
        query_names = [query_name_1, query_name_2]
        run_ids = list(self.metrics_dict[query_names[0]].keys())
        run_length = self.metrics_dict[query_names[0]][run_ids[0]].shape[0]
        if exclude_initial:
            start_index = 1
            significants = [False]
        else:
            start_index = 0
            significants = []
        for i in range(start_index, run_length):
            metrics_query_type_1 = []
            metrics_query_type_2 = []
            print(i)
            for run_id in run_ids:
                metrics_query_type_1.append(self.metrics_dict[query_name_1][run_id][i])
                metrics_query_type_2.append(self.metrics_dict[query_name_2][run_id][i])
            res = wilcoxon(metrics_query_type_1, metrics_query_type_2, alternative="less")
            if res.pvalue < 0.05:
                significants.append(True)
            else:
                significants.append(False)
        self.test_results[query_name_2] = significants

    def calculate_statistics_for_query(self, query_name):
        n = len(self.metrics_dict[query_name])
        metric_array_list = self.dict_to_list(self.metrics_dict[query_name])

        self.complete_metric_array_dict[query_name] = np.array(metric_array_list)
        print(query_name)

        mean_array = np.mean(metric_array_list, axis=0)
        std_array = np.std(metric_array_list, axis=0)
        median_array = np.median(metric_array_list, axis=0)
        print(metric_array_list[1])
        lower_quantile = np.quantile(np.array(metric_array_list), 0.25, axis=0)
        print(lower_quantile)
        upper_quantile = np.quantile(np.array(metric_array_list), 0.75, axis=0)
        quantile = norm.ppf(1 - self.alpha / 2)
        upper_ki = mean_array + (std_array / np.sqrt(n)) * quantile
        lower_ki = mean_array - (std_array / np.sqrt(n)) * quantile
        self.means_dict[query_name] = mean_array
        self.sigma_dict[query_name] = std_array
        self.upper_ki_dict[query_name] = upper_ki
        self.lower_ki_dict[query_name] = lower_ki
        self.median_dict[query_name] = median_array
        self.lower_quantile_dict[query_name] = lower_quantile
        self.upper_quantile_dict[query_name] = upper_quantile

    def calculate_statistics(self):
        for query_name in self.metrics_dict:
            self.calculate_statistics_for_query(query_name)

    def count_safety_violations(
        self, max_unsafe_allowed=0, only_include_query_names=False, query_names=None, use_ax=False, ax=None
    ):
        colors = []
        if not only_include_query_names:
            query_names = list(self.metrics_dict.keys())
        n = len(self.metrics_dict[query_names[0]])
        print(n)
        number_violated_list = []
        label_names = []
        for query_name in query_names:
            if query_name in self.color_dict:
                color = self.color_dict[query_name]
            label_name = self.name_dict[query_name]
            label_names.append(label_name)
            colors.append(color)
            number_violated = self.count_safety_violations_for_query(query_name, max_unsafe_allowed)
            number_violated_list.append(number_violated)
        if not use_ax:
            fig, ax = plt.subplots(figsize=(len(query_names) * 3, 9))
        ax.bar(label_names, number_violated_list, color=colors)
        ax.set_ylim(top=n)
        ax.set_yticks(np.arange(start=0, stop=n, step=3))
        title = "Number of runs with \n unsafe queries>=" + str(max_unsafe_allowed)

        ax.set_title(title)
        if not use_ax:
            plt.show()

    def create_safety_violations_sequence_plot(
        self, max_max_unsafe_allowed=3, only_include_query_names=False, query_names=None
    ):
        fig, axs = plt.subplots(1, max_max_unsafe_allowed, figsize=(7 * max_max_unsafe_allowed, 6))
        for i in range(0, max_max_unsafe_allowed):
            self.count_safety_violations(
                max_unsafe_allowed=i,
                only_include_query_names=only_include_query_names,
                query_names=query_names,
                use_ax=True,
                ax=axs[i],
            )
        plt.show()

    def count_safety_violations_for_query(self, query_name, max_unsafe_allowed):
        metric_array_list = self.dict_to_list(self.metrics_dict[query_name])
        number_violated = 0
        for metric_array in metric_array_list:
            n_unsafe_in_run = np.max(metric_array)
            if n_unsafe_in_run > max_unsafe_allowed:
                number_violated += 1
        return number_violated

    def create_plot(
        self,
        plot_type,
        save_fig,
        save_path,
        plot_name,
        metric_name,
        only_include_query_names=False,
        query_names=None,
        log_scale=False,
        use_subindexes=False,
        subindex_factor=10,
        bound_indexes=False,
        index_bound=300,
        add_first_index=False,
    ):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        counter = 0
        legend_patches = []
        if not only_include_query_names:
            query_names = list(self.metrics_dict.keys())
        for query_name in query_names:
            if query_name in self.color_dict:
                color = self.color_dict[query_name]
            else:
                color = self.color_list[counter]
                counter += 1
            indexes = list(range(0, self.means_dict[query_name].shape[0]))
            if use_subindexes:
                index_mask = [(index + 1) % subindex_factor == 0 for index in indexes]
                if add_first_index:
                    index_mask[0] = True
                # print(index_mask)
                indexes = np.array(indexes)[index_mask]

            if bound_indexes:
                indexes = indexes[indexes <= index_bound]
            if query_name in self.name_dict:
                label = self.name_dict[query_name]
            else:
                label = query_name

            if plot_type == PlotType.CONFIDENCE_BAND:
                ax.plot(indexes, self.means_dict[query_name], color=color, label=label)
                ax.fill_between(
                    indexes, self.lower_ki_dict[query_name], self.upper_ki_dict[query_name], color=color, alpha=0.05
                )
                ax.plot(indexes, self.lower_ki_dict[query_name], "--", color=color, alpha=0.3)
                ax.plot(indexes, self.upper_ki_dict[query_name], "--", color=color, alpha=0.3)
            elif plot_type == PlotType.CONFIDENCE_INTERVALLS:
                if use_subindexes:
                    ax.plot(indexes, self.means_dict[query_name][indexes], color=color, label=label)
                    ax.errorbar(
                        indexes,
                        self.means_dict[query_name][indexes],
                        yerr=self.upper_ki_dict[query_name][indexes] - self.lower_ki_dict[query_name][indexes],
                        marker="s",
                        color=color,
                        capsize=4,
                    )
                else:
                    ax.plot(indexes, self.means_dict[query_name], color=color, label=label)
                    ax.errorbar(
                        indexes,
                        self.means_dict[query_name],
                        yerr=self.upper_ki_dict[query_name] - self.lower_ki_dict[query_name],
                        marker="s",
                        color=color,
                        capsize=4,
                    )
            elif plot_type == PlotType.MEDIAN_INTERVAL:
                if use_subindexes:
                    errors = np.array(
                        [
                            self.median_dict[query_name][indexes] - self.lower_quantile_dict[query_name][indexes],
                            self.upper_quantile_dict[query_name][indexes] - self.median_dict[query_name][indexes],
                        ]
                    )
                    print(errors)
                    ax.plot(indexes, self.median_dict[query_name][indexes], color=color, label=label)
                    ax.errorbar(
                        indexes, self.median_dict[query_name][indexes], yerr=errors, marker="s", color=color, capsize=4
                    )
                else:
                    errors = np.array(
                        [
                            self.median_dict[query_name] - self.lower_quantile_dict[query_name],
                            self.upper_quantile_dict[query_name] - self.median_dict[query_name],
                        ]
                    )
                    ax.plot(indexes, self.median_dict[query_name], color=color, label=label)
                    ax.errorbar(
                        indexes,
                        self.median_dict[query_name],
                        yerr=[self.lower_quantile_dict[query_name], self.upper_quantile_dict[query_name]],
                        marker="s",
                        color=color,
                        capsize=4,
                    )

            elif plot_type == PlotType.MEAN:
                ax.plot(indexes, self.means_dict[query_name], color=color, label=label)
            elif plot_type == PlotType.BOX_PLOTS:
                boxprops = dict(linestyle="-", linewidth=2.0, edgecolor=color, facecolor=color)
                flierprops = dict(marker="o", markeredgecolor=color, markersize=4, linestyle="none")
                # medianprops = dict(marker='x',markerfacecolor=color,markeredgecolor=color)
                meanpointprops = dict(marker="D", markeredgecolor="black", markerfacecolor=color)
                whiskerprops = dict(linestyle="-", linewidth=2.0, color=color)
                if use_subindexes:
                    ax.boxplot(
                        self.complete_metric_array_dict[query_name][:, indexes],
                        patch_artist=True,
                        boxprops=boxprops,
                        meanprops=meanpointprops,
                        showmeans=False,
                        flierprops=flierprops,
                        manage_ticks=False,
                        whiskerprops=whiskerprops,
                        positions=indexes,
                    )
                else:
                    ax.boxplot(
                        self.complete_metric_array_dict[query_name],
                        patch_artist=True,
                        boxprops=boxprops,
                        meanprops=meanpointprops,
                        showmeans=False,
                        flierprops=flierprops,
                        manage_ticks=False,
                        whiskerprops=whiskerprops,
                        positions=indexes,
                    )

                    ax.set_xticks(np.arange(start=0, stop=indexes[-1] + 1, step=3))

                patch = mpatches.Patch(color=color, label=label)
                legend_patches.append(patch)

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel(metric_name + " (log-scale)")
        else:
            ax.set_ylabel(metric_name)
        ax.set_xlabel("Iteration")

        if plot_type == PlotType.BOX_PLOTS:
            ax.legend(handles=legend_patches)
        else:
            ax.legend()
        ax.set_title(metric_name)
        min_y, _ = ax.get_ylim()
        # ax.set_ylim((min_y,0.95))
        # ax.set_ylim((min_y,1.25))
        if not log_scale:
            min_y = min_y - 0.02
        if self.include_test:
            for query_name in self.test_results:
                if query_name in self.color_dict:
                    color = self.color_dict[query_name]
                else:
                    color = self.color_list[counter]
                for i, is_significant in enumerate(self.test_results[query_name]):
                    if is_significant:
                        if i % 3 == 0:
                            ax.plot([i], [min_y], "x", color=color, markersize=3)
                if log_scale:
                    min_y = min_y * 1.05
                else:
                    min_y = min_y + 0.01
        if save_fig:
            plt.savefig(os.path.join(save_path, plot_name + ".png"))
        else:
            plt.show()
