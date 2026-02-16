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

from typing import Union
from alef.acquisition_function.acquisition_function_factory import AcquisitionFunctionFactory
from alef.bayesian_optimization.bayesian_optimizer import BayesianOptimizer
from alef.bayesian_optimization.bayesian_optimizer_objects import BayesianOptimizerObjects

from alef.configs.bayesian_optimization.bayesian_optimizer_configs import BaseBOConfig
from alef.configs.bayesian_optimization.bayesian_optimizer_objects_configs import BaseObjectBOConfig


class BayesianOptimizerFactory:
    @staticmethod
    def build(bayesian_optimizer_config: Union[BaseBOConfig, BaseObjectBOConfig]):
        """
        Build a Bayesian optimizer based on the configuration.
        """
        if isinstance(bayesian_optimizer_config, BaseBOConfig):
            acquisition_function_config = bayesian_optimizer_config.acquisition_function_config
            acquisition_function = AcquisitionFunctionFactory.build(acquisition_function_config)
            return BayesianOptimizer(acquisition_function=acquisition_function, **bayesian_optimizer_config.dict())
        elif isinstance(bayesian_optimizer_config, BaseObjectBOConfig):
            acquisition_function_config = bayesian_optimizer_config.acquisition_function_config
            acquisition_function = AcquisitionFunctionFactory.build(acquisition_function_config)
            return BayesianOptimizerObjects(
                acquisition_function=acquisition_function, **bayesian_optimizer_config.dict()
            )
