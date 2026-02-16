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
import torch
from typing import List, Optional
from pathlib import Path
from alef.active_learner.amortized_policies.amortized_policy_factory import AmortizedPolicyFactory
from alef.configs.active_learner.amortized_policies.policy_configs import ContinuousGPPolicyConfig
from alef.enums.active_learner_enums import ValidationType
from alef.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot

from .base_oracle_active_learner import BaseOracleActiveLearner


class ContinuousPolicyActiveLearner(BaseOracleActiveLearner):
    """
    Main class for non-batch oracle-based active learning - one query at a time - collects queries by calling its oracle object

    Main Attributes:
        validation_type : ValidationType - Enum which validation metric should be used e.g. RMSE, NegLoglikeli
        acquisiton_optimization_type : OracleALAcquisitionOptimizationType - specifies how the acquisition function should be optimized
    """

    def __init__(
        self,
        validation_type: ValidationType,
        policy_path: str = "",
        validation_at: Optional[List[int]] = None,
        *,
        pytest: bool = False,  # use this only to do pytest
        policy_dimension: int = 2,  # specify only to do pytest
        **kwargs,
    ):
        """
        Initialize the ContinuousPolicyActiveLearner.

        Args:
            validation_type (ValidationType): The type of validation to perform.
            policy_path (str, optional): Path to the policy. Defaults to ''.
            validation_at (Optional[List[int]], optional): The steps at which to perform validation. Defaults to None.
            pytest (bool, optional): Whether to use pytest. Defaults to False.
            policy_dimension (int, optional): The dimension of the policy. Defaults to 2.
        """
        super().__init__()
        self.validation_type = validation_type
        self.validation_at = validation_at
        self.initialize_validation_metrics()
        if pytest:
            self.load_test_policy(policy_dimension)
        else:
            self.load_policy(policy_path)

    def load_policy(self, path):
        """
        Load the policy from the specified path.

        Args:
            path (str): The path to the policy.
        """
        root_path = Path(path)
        # metric_path = root_path / 'check_point_overview.xlsx'
        # assert metric_path.exists()
        # df = pd.read_excel(metric_path, index_col=0)
        # df = df.sort_values(by='rmse_mean', ascending=True)
        # model_path = root_path / 'result' / f'model_checkpoint_{df.index[0]}.pth'
        model_path = root_path / "result" / "model_checkpoint_final.pth"
        # assert config_path.exists()
        print(f"Load model: {model_path}")
        assert model_path.exists()
        # with open(config_path, mode='r') as json_file:
        # config_dict = json.load(json_file)
        D = int(path.split("ContinuousGP")[-1].split("DPolicy")[0])
        policy_config = ContinuousGPPolicyConfig(input_dim=D, self_attention_layer=True)
        self.policy = AmortizedPolicyFactory.build(policy_config)
        model_state_dict = torch.load(model_path)
        self.policy.load_state_dict(model_state_dict)
        self.policy.eval()

    def load_test_policy(self, policy_dimension: int):
        """
        Load a test policy for pytest.

        Args:
            policy_dimension (int): The dimension of the policy.
        """
        D = policy_dimension
        policy_config = ContinuousGPPolicyConfig(input_dim=D, self_attention_layer=True)
        self.policy = AmortizedPolicyFactory.build(policy_config)
        self.policy.eval()

    def run_policy(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Run the policy to get the next query point.

        Args:
            x_data (np.ndarray): The input data.
            y_data (np.ndarray): The output data.

        Returns:
            np.ndarray: The next query point.
        """
        N = x_data.shape[0]
        x_list = [torch.from_numpy(x_data[i, None, :]).to(torch.get_default_dtype()) for i in range(N)]
        y_list = [torch.from_numpy(y_data[i, :]).to(torch.get_default_dtype()) for i in range(N)]

        query = self.policy(*zip(x_list, y_list)).cpu().detach().numpy()

        return query[0, :]

    def update_gp_model(self):
        """
        Update the Gaussian Process model with the current data.
        """
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)

    def update(self):
        """
        Update the active learner by selecting a new query point.

        Returns:
            np.ndarray: The selected query point.
        """
        assert np.allclose(self.policy.input_domain, self.oracle.get_box_bounds())
        return self.run_policy(self.x_data, self.y_data)

    def learn(self, n_steps: int):
        """
        Main Active learning loop - makes n_steps queries (calls the self.oracle object at each query) and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        if self.do_plotting and self.oracle.get_dimension() <= 2:
            self.sample_ground_truth()
        self.n_steps = n_steps
        for i in range(0, self.n_steps):
            gp_updated = False
            query = self.update()
            print("Query")
            print(query)
            new_y = self.oracle.query(query)
            if self.do_plotting and self.oracle.get_dimension() <= 2:
                if not gp_updated:
                    self.update_gp_model()
                    gp_updated = True
                self.plot(query, new_y, i)
            self.add_train_point(i, query, new_y)
            if self.validation_at is None or len(self.validation_at) == 0 or i in self.validation_at:
                if not gp_updated:
                    self.update_gp_model()
                    gp_updated = True
                self.validate(i)
        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def validate(self, idx: int):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used

        Args:
            idx (int): The step index.
        """
        if self.validation_type == ValidationType.RMSE:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test)
            rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
            self.add_validation_value(idx, rmse)
        elif self.validation_type == ValidationType.NEG_LOG_LIKELI:
            log_likelis = self.model.predictive_log_likelihood(self.x_test, self.y_test)
            neg_log_likeli = np.mean(-1 * log_likelis)
            self.add_validation_value(idx, neg_log_likeli)

    def plot(self, query: np.array, new_y: np.float, step: int):
        """
        Plot the results of the active learning process.

        Args:
            query (np.array): The query made by the active learner.
            new_y (np.float): The new data point obtained.
            step (int): The current step of the active learning process.
        """
        dimension = self.oracle.get_dimension()
        x_grid = self.gt_X
        y_over_grid = self.gt_function_values
        if dimension == 1:
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            if self.save_plots:
                plot_name = "query_" + str(step) + ".png"
                active_learning_1d_plot(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    True,
                    self.gt_X,
                    self.gt_function_values,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                active_learning_1d_plot(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    True,
                    self.gt_X,
                    self.gt_function_values,
                )
        elif dimension == 2:
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            acquisition_function = pred_sigma
            if self.save_plots:
                plot_name = "query_" + str(step) + ".png"
                active_learning_2d_plot(
                    x_grid,
                    acquisition_function,
                    pred_mu,
                    y_over_grid,
                    self.x_data,
                    query,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                active_learning_2d_plot(x_grid, acquisition_function, pred_mu, y_over_grid, self.x_data, query)
        else:
            print("Dimension too high for plotting")


if __name__ == "__main__":
    pass
