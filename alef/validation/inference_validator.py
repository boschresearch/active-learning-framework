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
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from alef.utils.metric_curve_plotter import MetricCurvePlotter
import alef.validation.plot_dicts as plot_dicts


class InferenceValidator:
    def __init__(self, base_folder: str):
        self.metrics_dict = {}
        self.summary_statistics_dict = {}
        self.base_folder = base_folder
        self.metric_name: str
        self.for_kernel_kernel_inference = True

    def collect_metrics(
        self,
        metric_name_filter="RMSE",
        parallel_files=False,
        use_index_in_file=False,
        index_in_file=0,
        use_only_first_metrics=False,
        n_first=5,
    ):
        self.metric_name = metric_name_filter
        sub_dirs = os.listdir(self.base_folder)
        print(sub_dirs)
        for sub_dir in sub_dirs:
            run_name = sub_dir
            run_path = os.path.join(self.base_folder, run_name)
            if parallel_files:
                assert not self.for_kernel_kernel_inference
                assert not use_only_first_metrics
                self.metrics_dict[run_name] = self.collect_parallel_run_metrics(run_path, metric_name_filter)
            else:
                self.metrics_dict[run_name] = self.collect_run_metrics(
                    run_path, metric_name_filter, use_index_in_file, index_in_file, use_only_first_metrics, n_first
                )

    def collect_parallel_run_metrics(self, run_path: str, metric_name_filter: str):
        file_names = [file_name for file_name in os.listdir(run_path) if file_name.endswith(".txt")]
        run_dict = {}
        for file_name in file_names:
            dataset_name, n_train, n_test, index, metric_name = self.extract_meta_info_from_parallel_file_name(
                file_name
            )
            if metric_name == metric_name_filter:
                if not dataset_name in run_dict:
                    run_dict[dataset_name] = {}
                if not n_train in run_dict[dataset_name]:
                    run_dict[dataset_name][n_train] = []
                metrics_array = np.loadtxt(os.path.join(run_path, file_name))
                run_dict[dataset_name][n_train].append(metrics_array)
        for dataset_name in run_dict:
            for n_train in run_dict[dataset_name]:
                metrics_list = run_dict[dataset_name][n_train]
                run_dict[dataset_name][n_train] = np.array(metrics_list)
        return run_dict

    def collect_run_metrics(
        self, run_path: str, metric_name_filter: str, use_index_in_file, index_in_file, use_only_first_metrics, n_first
    ) -> Dict:
        file_names = [file_name for file_name in os.listdir(run_path) if file_name.endswith(".txt")]
        run_dict = {}
        for file_name in file_names:
            if self.for_kernel_kernel_inference:
                dataset_name, n_train, n_test, metric_name = self.extract_meta_info_from_kernel_kernel_file_name(
                    file_name
                )
            else:
                dataset_name, n_train, n_test, metric_name = self.extract_meta_info_from_file_name(file_name)
            if metric_name == metric_name_filter:
                if not dataset_name in run_dict:
                    run_dict[dataset_name] = {}
                metrics_array = np.loadtxt(os.path.join(run_path, file_name))
                if use_index_in_file:
                    metrics_array = metrics_array[:, index_in_file]
                    print(metrics_array)
                if use_only_first_metrics:
                    print(dataset_name)
                    print(n_train)
                    print(run_path)
                    print(metrics_array[:n_first])
                    run_dict[dataset_name][n_train] = metrics_array[:n_first]
                else:
                    run_dict[dataset_name][n_train] = metrics_array
        return run_dict

    def extract_meta_info_from_file_name(self, file_name: str) -> Tuple[str, int, int]:
        file_name = file_name.split(".")[0]
        data_set_name = file_name.split("_")[0]
        n_train = file_name.split("_")[2]
        n_test = file_name.split("_")[4]
        metric_name = file_name.split("_")[5]
        return data_set_name, int(n_train), int(n_test), metric_name

    def extract_meta_info_from_kernel_kernel_file_name(self, file_name: str) -> Tuple[str, int, int]:
        file_name = file_name.split(".")[0]
        data_set_name = file_name.split("_")[0]
        target_name = file_name.split("_")[3]
        search_space_name = file_name.split("_")[5]
        n_train = file_name.split("_")[7]
        n_test = file_name.split("_")[9]
        metric_name = file_name.split("_")[-1]
        dataset_identifier = data_set_name + "_" + target_name + "_" + search_space_name
        return dataset_identifier, int(n_train), int(n_test), metric_name

    def extract_meta_info_from_parallel_file_name(self, file_name: str) -> Tuple[str, int, int]:
        file_name = file_name.split(".")[0]
        data_set_name = file_name.split("_")[0]
        n_train = file_name.split("_")[2]
        n_test = file_name.split("_")[4]
        index = file_name.split("_")[6]
        metric_name = file_name.split("_")[7]
        return data_set_name, int(n_train), int(n_test), index, metric_name

    def create_summary_dict(self):
        for run_name in self.metrics_dict:
            if not run_name in self.summary_statistics_dict:
                self.summary_statistics_dict[run_name] = {}
            for data_set in self.metrics_dict[run_name]:
                if not data_set in self.summary_statistics_dict[run_name]:
                    self.summary_statistics_dict[run_name][data_set] = {}
                for n_train in self.metrics_dict[run_name][data_set]:
                    if not n_train in self.summary_statistics_dict[run_name][data_set]:
                        self.summary_statistics_dict[run_name][data_set][n_train] = {}
                    metrics_array = self.metrics_dict[run_name][data_set][n_train]
                    self.summary_statistics_dict[run_name][data_set][n_train]["median"] = np.median(metrics_array)
                    self.summary_statistics_dict[run_name][data_set][n_train]["std"] = np.std(metrics_array)
                    self.summary_statistics_dict[run_name][data_set][n_train]["mean"] = np.mean(metrics_array)
                    self.summary_statistics_dict[run_name][data_set][n_train]["lower_quartile"] = np.quantile(
                        metrics_array, 0.25
                    )
                    self.summary_statistics_dict[run_name][data_set][n_train]["upper_quartile"] = np.quantile(
                        metrics_array, 0.75
                    )
                    self.summary_statistics_dict[run_name][data_set][n_train]["n"] = len(metrics_array)

    def create_summary_statistics_table(
        self, run_list=[], use_run_list=False, exclude_list=[], summary_statistic_name="median", mark_min_values=True
    ):
        self.create_summary_dict()
        if use_run_list:
            run_names = run_list
        else:
            run_names = list(self.metrics_dict.keys())
        assert len(run_names) > 0
        summary_statistics_dfs = []
        col_name_list = []
        for run_name in run_names:
            if not run_name in exclude_list:
                summary_statistics_dict_complete = {}
                for dataset in self.metrics_dict[run_name]:
                    n_train_list = list(self.metrics_dict[run_name][dataset].keys())
                    n_train_list.sort()
                    for n_train in n_train_list:
                        col_name = dataset + " - n=" + str(n_train)
                        col_name_list.append(col_name)
                        summary_statistic = self.summary_statistics_dict[run_name][dataset][n_train][
                            summary_statistic_name
                        ]
                        summary_statistics_dict_complete[col_name] = [summary_statistic]
                run_name_readable = self.run_name_to_human_readable(run_name)
                median_data_frame = pd.DataFrame(data=summary_statistics_dict_complete, index=[run_name_readable])
                summary_statistics_dfs.append(median_data_frame)
        final_df = pd.concat(summary_statistics_dfs)
        self.print_latex_tables(final_df, mark_min_values)

    def print_latex_tables(self, data_frame, mark_min_values=True):
        if mark_min_values:
            for col in data_frame.columns.get_level_values(0).unique():
                data = data_frame[col]
                extrema = data != data.min()
                format_string = "%.5f"
                bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
                formatted = data.apply(lambda x: format_string % x)
                data_frame[col] = formatted.where(extrema, bolded)
        print(data_frame.transpose().to_latex(escape=False))

    def create_plots(
        self,
        run_list=[],
        use_run_list=False,
        include_error_bar=False,
        use_different_metric_name=False,
        new_metric_name="",
    ):
        self.create_summary_dict()
        data_sets = list(self.summary_statistics_dict[list(self.summary_statistics_dict.keys())[0]].keys())
        plotter = MetricCurvePlotter(len(data_sets))
        colors = ["red", "orange", "green", "navy", "brown", "grey", "black"]
        if use_run_list:
            run_names = run_list
        else:
            run_names = list(self.summary_statistics_dict.keys())
        assert len(run_names) > 0

        for i, data_set in enumerate(data_sets):
            for j, run_name in enumerate(run_names):
                if data_set in self.summary_statistics_dict[run_name]:
                    n_train_list = list(self.metrics_dict[run_name][data_set].keys())
                    # n_train_list = [100, 200, 500, 700]
                    n_train_list.sort()
                    n_train_list = n_train_list[1:]
                    summary_statistics_list = []
                    error_lower = []
                    error_upper = []
                    for n_train in n_train_list:
                        summary_statistic = self.summary_statistics_dict[run_name][data_set][n_train]["median"]
                        summary_statistics_list.append(summary_statistic)
                        lower_quartile = self.summary_statistics_dict[run_name][data_set][n_train]["lower_quartile"]
                        upper_quartile = self.summary_statistics_dict[run_name][data_set][n_train]["upper_quartile"]
                        error_lower.append(lower_quartile)
                        error_upper.append(upper_quartile)
                    if include_error_bar:
                        plotter.add_metrics_curve_with_errors(
                            n_train_list,
                            summary_statistics_list,
                            error_lower,
                            error_upper,
                            colors[j],
                            self.run_name_to_human_readable(run_name),
                            i,
                        )
                    else:
                        plotter.add_metrics_curve(
                            n_train_list,
                            summary_statistics_list,
                            colors[j],
                            self.run_name_to_human_readable(run_name),
                            i,
                        )
                    if use_different_metric_name:
                        plotter.configure_axes(i, data_set, "n_data", new_metric_name, False, True)
                    else:
                        plotter.configure_axes(i, data_set, "n_data", self.metric_name, False, True)

        plotter.show()
        # for run_name in self.summary_statistics_dict:

    def run_name_to_human_readable(self, run_name: str):
        if run_name in plot_dicts.name_dict:
            return plot_dicts.name_dict[run_name]
        else:
            return run_name
