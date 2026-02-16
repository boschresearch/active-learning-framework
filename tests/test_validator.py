from alef.validation.advanced_validator import SingleRun
import numpy as np


def test_single_run_in_advanced_validator():
    run = SingleRun("Run1", 0)
    iteration_times = np.array([10.0, 20.0, 10.0])
    metric_values = np.array([-1.0, 1.0, 1.5, 1.6])
    result_time_array = np.arange(0, 25, 5)
    result_metric_array = np.array([-1.0, 0.0, 1.0, 1.125, 1.25])
    run.set_iteration_time(iteration_times)
    run.set_main_metric(metric_values)
    metric_over_time, time_array, _ = run.get_metric_array_over_time_array(5.0, 20)
    assert np.allclose(metric_over_time, result_metric_array)
    assert np.allclose(time_array, result_time_array)


if __name__ == "__main__":
    test_single_run_in_advanced_validator()
