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

from alef.configs.kernels.kernel_list_configs import BasicKernelListConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import (
    BasicLinearKernelPytorchConfig,
    BasicPeriodicKernelPytorchConfig,
    BasicRBFPytorchConfig,
    BasicRQKernelPytorchConfig,
    BasicMatern32PytorchConfig,
    BasicMatern52PytorchConfig,
)
from alef.configs.kernels.pytorch_kernels.spectral_mixture_kernel_configs import BasicSpectralMixturePytorchConfig
from alef.configs.kernels.pytorch_kernels.hhk_pytorch_configs import BasicHHKPytorchConfig
from alef.kernels.pytorch_kernels.elementary_kernels_pytorch import (
    LinearKernelPytorch,
    PeriodicKernelPytorch,
    RBFKernelPytorch,
    RQKernelPytorch,
    Matern32KernelPytorch,
    Matern52KernelPytorch,
)
from alef.kernels.pytorch_kernels.spetral_mixture_kernel_pytorch import SpectralMixtureKernelPytorch
from alef.kernels.pytorch_kernels.hierarchical_hyperplane_kernel_pytorch import HierarchicalHyperplaneKernelPytorch


class PytorchKernelFactory:
    @staticmethod
    def build(kernel_config: BaseKernelPytorchConfig):
        if isinstance(kernel_config, BasicRBFPytorchConfig):
            return RBFKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicPeriodicKernelPytorchConfig):
            return PeriodicKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicRQKernelPytorchConfig):
            return RQKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicLinearKernelPytorchConfig):
            return LinearKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicMatern52PytorchConfig):
            return Matern52KernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicMatern32PytorchConfig):
            return Matern32KernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicKernelListConfig):
            from alef.models.amortized_infer_structured_kernels.gp.base_kernels import (
                transform_kernel_list_to_expression,
            )

            expanded_kernel_list = [kernel_config.kernel_list for i in range(0, kernel_config.input_dimension)]
            kernel_expression = transform_kernel_list_to_expression(
                expanded_kernel_list, add_prior=kernel_config.add_prior, use_gpflow=False
            )
            return kernel_expression.get_kernel()
        elif isinstance(kernel_config, BasicSpectralMixturePytorchConfig):
            return SpectralMixtureKernelPytorch(**kernel_config.dict())
        elif isinstance(kernel_config, BasicHHKPytorchConfig):
            base_kernel_config = kernel_config.base_kernel_config
            base_kernel_config.input_dimension = kernel_config.input_dimension
            base_kernel = PytorchKernelFactory.build(base_kernel_config)
            return HierarchicalHyperplaneKernelPytorch(base_kernel=base_kernel, **kernel_config.dict())
        else:
            raise NotImplementedError(f"invalid kernel config {kernel_config}")
