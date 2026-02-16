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

from alef.models.amortized_infer_structured_kernels.config.nn.kernel_encoder_configs import (
    BaseKernelEncoderConfig,
    CrossAttentionKernelEncoderConfig,
    KernelEncoderConfig,
)
from alef.models.amortized_infer_structured_kernels.nn.kernel_encoder import CrossAttentionKernelEncoder, KernelEncoder


class KernelEncoderFactory:
    def build(kernel_encoder_config: BaseKernelEncoderConfig):
        if isinstance(kernel_encoder_config, KernelEncoderConfig):
            kernel_encoder = KernelEncoder(**kernel_encoder_config.dict())
            return kernel_encoder
        elif isinstance(kernel_encoder_config, CrossAttentionKernelEncoderConfig):
            kernel_encoder = CrossAttentionKernelEncoder(**kernel_encoder_config.dict())
            return kernel_encoder
        else:
            raise ValueError
