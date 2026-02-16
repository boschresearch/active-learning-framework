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

"""
Prior parameters for kernel parameters - define globally to
assert consitency between implementations (gpflow,gpytorch,amortized_inference) adn between
parts and combined kernels such as CH operator and HHK kernel etc.
"""

from enum import IntEnum
import numpy as np


class PriorSettings(IntEnum):
    STANDARD = 1
    HIGH_NOISE = 2
    FLEXIBLE_ZERO_ONE = 3


PRIOR_SETTING = PriorSettings.STANDARD

if PRIOR_SETTING == PriorSettings.STANDARD:
    EXPECTED_OBSERVATION_NOISE = 0.1
    NOISE_VARIANCE_EXPONENTIAL_LAMBDA = 1.0 / np.power(EXPECTED_OBSERVATION_NOISE, 2.0)
    KERNEL_VARIANCE_GAMMA = (2.0, 3.0)
    KERNEL_LENGTHSCALE_GAMMA = (2.0, 2.0)
    LINEAR_KERNEL_OFFSET_GAMMA = (2.0, 3.0)
    RQ_KERNEL_ALPHA_GAMMA = (2.0, 2.0)
    PERIODIC_KERNEL_PERIOD_GAMMA = (2.0, 2.0)
    HHK_SMOOTHING_PRIOR_GAMMA = (6.0, 2.0)
    WEIGHTED_ADDITIVE_KERNEL_ALPHA_NORMAL = (0.0, 2.0)

if PRIOR_SETTING == PriorSettings.HIGH_NOISE:
    EXPECTED_OBSERVATION_NOISE = 0.5
    NOISE_VARIANCE_EXPONENTIAL_LAMBDA = 1.0 / np.power(EXPECTED_OBSERVATION_NOISE, 2.0)
    KERNEL_VARIANCE_GAMMA = (10.0, 9.0)
    KERNEL_LENGTHSCALE_GAMMA = (2.0, 2.0)
    LINEAR_KERNEL_OFFSET_GAMMA = (2.0, 3.0)
    RQ_KERNEL_ALPHA_GAMMA = (2.0, 2.0)
    PERIODIC_KERNEL_PERIOD_GAMMA = (2.0, 2.0)
    HHK_SMOOTHING_PRIOR_GAMMA = (6.0, 2.0)
    WEIGHTED_ADDITIVE_KERNEL_ALPHA_NORMAL = (0.0, 2.0)

elif PRIOR_SETTING == PriorSettings.FLEXIBLE_ZERO_ONE:
    EXPECTED_OBSERVATION_NOISE = 0.15
    NOISE_VARIANCE_EXPONENTIAL_LAMBDA = 1.0 / np.power(EXPECTED_OBSERVATION_NOISE, 2.0)
    KERNEL_VARIANCE_GAMMA = (2.0, 3.0)
    KERNEL_LENGTHSCALE_GAMMA = (2.0, 5.0)
    LINEAR_KERNEL_OFFSET_GAMMA = (2.0, 3.0)
    RQ_KERNEL_ALPHA_GAMMA = (2.0, 2.0)
    PERIODIC_KERNEL_PERIOD_GAMMA = (2.0, 3.0)
    HHK_SMOOTHING_PRIOR_GAMMA = (6.0, 2.0)
    WEIGHTED_ADDITIVE_KERNEL_ALPHA_NORMAL = (0.0, 2.0)
