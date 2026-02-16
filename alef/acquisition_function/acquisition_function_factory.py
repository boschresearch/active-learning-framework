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
from alef.acquisition_function.al_acquisition_functions.pred_entropy_batch import PredEntropyBatch
from alef.acquisition_function.bo_acquisition_functions.integrated_ei import IntegratedEI
from alef.configs.acquisition_function.al_acquisition_functions.base_al_acquisition_function_config import (
    BaseALAcquisitionFunctionConfig,
)
from alef.configs.acquisition_function.bo_acquisition_functions.base_bo_acquisition_function_config import (
    BaseBOAcquisitionFunctionConfig,
)
from alef.configs.acquisition_function.al_acquisition_functions.pred_entropy_batch_config import (
    BasicPredEntropyBatchConfig,
)
from alef.configs.acquisition_function.safe_acquisition_functions.base_safe_acquisition_function_config import (
    BaseSafeAcquisitionFunctionConfig,
)

from alef.configs.acquisition_function.al_acquisition_functions.acq_random_config import BasicRandomConfig
from .al_acquisition_functions.acq_random import Random
from alef.configs.acquisition_function.safe_acquisition_functions.safe_random_config import BasicSafeRandomConfig
from .safe_acquisition_functions.safe_random import SafeRandom
from alef.configs.acquisition_function.al_acquisition_functions.pred_variance_config import BasicPredVarianceConfig
from alef.configs.acquisition_function.al_acquisition_functions.pred_sigma_config import BasicPredSigmaConfig
from .al_acquisition_functions.pred_variance import PredVariance
from alef.configs.acquisition_function.al_acquisition_functions.pred_entropy_config import BasicPredEntropyConfig
from .al_acquisition_functions.pred_entropy import PredEntropy
from .al_acquisition_functions.pred_sigma import PredSigma
from alef.configs.acquisition_function.safe_acquisition_functions.safe_pred_entropy_config import (
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
)
from .safe_acquisition_functions.safe_pred_entropy import SafePredEntropy, SafePredEntropyAll
from alef.configs.acquisition_function.safe_acquisition_functions.safe_discover_config import (
    BasicSafeDiscoverConfig,
    BasicSafeDiscoverQuantileConfig,
)
from .safe_acquisition_functions.safe_discover import SafeDiscover, SafeDiscoverQuantile

from alef.configs.acquisition_function.safe_acquisition_functions.safe_opt_config import BasicSafeOptConfig
from .safe_acquisition_functions.safe_opt import SafeOpt
from alef.configs.acquisition_function.safe_acquisition_functions.safe_gp_ucb_config import BasicSafeGPUCBConfig
from .safe_acquisition_functions.safe_gp_ucb import SafeGPUCB
from alef.configs.acquisition_function.bo_acquisition_functions.ei_config import BasicEIConfig
from .bo_acquisition_functions.ei import EI
from alef.configs.acquisition_function.safe_acquisition_functions.safe_ei_config import BasicSafeEIConfig
from .safe_acquisition_functions.safe_ei import SafeEI
from alef.configs.acquisition_function.safe_acquisition_functions.safe_discover_opt_config import (
    BasicSafeDiscoverOptConfig,
    BasicSafeDiscoverOptQuantileConfig,
)
from .safe_acquisition_functions.safe_discover_opt import SafeDiscoverOpt, SafeDiscoverOptQuantile
from alef.configs.acquisition_function.bo_acquisition_functions.integrated_ei_config import BasicIntegratedEIConfig
from alef.acquisition_function.bo_acquisition_functions.gp_ucb import GPUCB
from alef.configs.acquisition_function.bo_acquisition_functions.gp_ucb_config import BasicGPUCBConfig


class AcquisitionFunctionFactory:
    @staticmethod
    def build(
        function_config: Union[
            BaseALAcquisitionFunctionConfig, BaseBOAcquisitionFunctionConfig, BaseSafeAcquisitionFunctionConfig
        ],
    ):
        if isinstance(function_config, BasicRandomConfig):
            return Random(**function_config.dict())
        elif isinstance(function_config, BasicSafeRandomConfig):
            return SafeRandom(**function_config.dict())
        elif isinstance(function_config, BasicPredVarianceConfig):
            return PredVariance(**function_config.dict())
        elif isinstance(function_config, BasicPredSigmaConfig):
            return PredSigma(**function_config.dict())
        elif isinstance(function_config, BasicPredEntropyConfig):
            return PredEntropy(**function_config.dict())
        elif isinstance(function_config, BasicSafePredEntropyConfig):
            return SafePredEntropy(**function_config.dict())
        elif isinstance(function_config, BasicSafePredEntropyAllConfig):
            return SafePredEntropyAll(**function_config.dict())
        elif isinstance(function_config, BasicSafeDiscoverConfig):
            return SafeDiscover(**function_config.dict())
        elif isinstance(function_config, BasicSafeDiscoverQuantileConfig):
            return SafeDiscoverQuantile(**function_config.dict())
        elif isinstance(function_config, BasicSafeOptConfig):
            return SafeOpt(**function_config.dict())
        elif isinstance(function_config, BasicSafeGPUCBConfig):
            return SafeGPUCB(**function_config.dict())
        elif isinstance(function_config, BasicEIConfig):
            return EI(**function_config.dict())
        elif isinstance(function_config, BasicIntegratedEIConfig):
            return IntegratedEI(**function_config.dict())
        elif isinstance(function_config, BasicGPUCBConfig):
            return GPUCB(**function_config.dict())
        elif isinstance(function_config, BasicSafeEIConfig):
            return SafeEI(**function_config.dict())
        elif isinstance(function_config, BasicSafeDiscoverOptConfig):
            return SafeDiscoverOpt(**function_config.dict())
        elif isinstance(function_config, BasicSafeDiscoverOptQuantileConfig):
            return SafeDiscoverOptQuantile(**function_config.dict())
        elif isinstance(function_config, BasicPredEntropyBatchConfig):
            return PredEntropyBatch(**function_config.dict())
        else:
            raise NotImplementedError("Invalid config")
