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

from .al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from .al_acquisition_functions.base_batch_al_acquisition_function import BaseBatchALAcquisitionFunction
from .bo_acquisition_functions.base_bo_acquisition_function import BaseBOAcquisitionFunction
from .safe_acquisition_functions.base_safe_acquisition_function import (
    BaseSafeAcquisitionFunction,
    StandardAlphaAcquisitionFunction,
    StandardBetaAcquisitionFunction,
)
from .al_acquisition_functions.acq_random import Random
from .al_acquisition_functions.pred_sigma import PredSigma
from .al_acquisition_functions.pred_variance import PredVariance
from .al_acquisition_functions.pred_entropy import PredEntropy
from .al_acquisition_functions.pred_entropy_batch import PredEntropyBatch

from .bo_acquisition_functions.ei import EI
from .bo_acquisition_functions.gp_ucb import GPUCB
from .bo_acquisition_functions.integrated_ei import IntegratedEI

from .safe_acquisition_functions.safe_random import SafeRandom
from .safe_acquisition_functions.safe_pred_entropy import SafePredEntropy, SafePredEntropyAll
from .safe_acquisition_functions.safe_discover import SafeDiscover, SafeDiscoverQuantile

from .safe_acquisition_functions.safe_ei import SafeEI
from .safe_acquisition_functions.safe_opt import SafeOpt
from .safe_acquisition_functions.safe_gp_ucb import SafeGPUCB
from .safe_acquisition_functions.safe_discover_opt import SafeDiscoverOpt, SafeDiscoverOptQuantile

__all__ = [
    "BaseALAcquisitionFunction",
    "BaseBatchALAcquisitionFunction",
    "BaseBOAcquisitionFunction",
    "BaseSafeAcquisitionFunction",
    "StandardAlphaAcquisitionFunction",
    "StandardBetaAcquisitionFunction",
    "Random",
    "SafeRandom",
    "PredSigma",
    "PredVariance",
    "PredEntropy",
    "PredEntropyBatch",
    "SafePredEntropy",
    "SafePredEntropyAll",
    "SafeDiscover",
    "SafeDiscoverQuantile",
    "EI",
    "IntegratedEI",
    "SafeEI",
    "SafeOpt",
    "GPUCB",
    "SafeGPUCB",
    "SafeDiscoverOpt",
    "SafeDiscoverOptQuantile",
]
