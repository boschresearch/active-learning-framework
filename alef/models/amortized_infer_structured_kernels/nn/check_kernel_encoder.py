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

from alef.models.amortized_infer_structured_kernels.utils.utils import get_test_inputs
from alef.models.amortized_infer_structured_kernels.nn.dataset_encoder import DatasetEncoder
from alef.models.amortized_infer_structured_kernels.nn.kernel_encoder import CrossAttentionKernelEncoder

(
    X_list,
    y_list,
    X_padded,
    y_padded,
    N,
    size_mask,
    dim_mask,
    kernel_embeddings,
    kernel_mask,
    size_mask_kernel,
    _,
) = get_test_inputs(2, 10, 4, 3, only_SE=False)
encoder = DatasetEncoder(num_attentions1=2, num_attentions2=2, att1_hidden_dim=16, att2_hidden_dim=16)
output = encoder.forward(X_padded, y_padded, size_mask, dim_mask)
kernel_encoder = CrossAttentionKernelEncoder(10, 16, 16, 4, 4, 4)
kernel_encoder.forward(kernel_embeddings, output, kernel_mask, dim_mask)
