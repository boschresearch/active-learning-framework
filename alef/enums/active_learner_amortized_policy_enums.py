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


class DomainWarpperType(Enum):
    SIGMOID = 1
    TANH = 2


class LossType(Enum):
    DAD = 1
    DAD_SCORE = 2
    GP_ENTROPY1 = 11
    GP_ENTROPY2 = 12
    GP_MI1 = 13
    GP_MI2 = 14
    GP_MI_ENTROPY1 = 101
    GP_MI_ENTROPY2 = 102


class GPSampleMethod(Enum):
    RFF = 1
    GP = 2
