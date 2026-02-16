import numpy as np
import tensorflow as tf
import torch

from alef.kernels.kernel_factory import KernelFactory
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig

from alef.utils.separable_cholesky import SeparableCholeskyDecomposition

Ns = 10
Nt = 10
D = 3
x = np.random.standard_normal([Ns + Nt, D])


def test_separable_cholesky_tensorflow():
    kernel = KernelFactory.build(RBFWithPriorConfig(input_dimension=D))

    cholesky_decomp = SeparableCholeskyDecomposition(kernel)

    cholesky_decomp._initialize_cholesky(x[:Ns], 0.1)

    K_ref = kernel(x, full_cov=True)
    L_ref = tf.linalg.cholesky(K_ref + 0.1 * tf.eye(Ns + Nt, dtype=K_ref.dtype)).numpy()
    cholesky_decomp._update_cholesky(x[Ns:], 0.1)
    assert np.allclose(L_ref, cholesky_decomp._decomposer._L.numpy(), atol=1e-5)
    cholesky_decomp = SeparableCholeskyDecomposition(kernel)
    cholesky_decomp.new_cholesky(x[:Ns], 0.1)
    assert np.allclose(L_ref, cholesky_decomp.new_cholesky(x[Ns:], 0.1).numpy(), atol=1e-5)


def test_separable_cholesky_torch():
    kernel = PytorchKernelFactory.build(RBFWithPriorPytorchConfig(input_dimension=D))

    cholesky_decomp = SeparableCholeskyDecomposition(kernel)

    xx = torch.from_numpy(x).to(torch.get_default_dtype())
    cholesky_decomp._initialize_cholesky(xx[:Ns], 0.1)

    K_ref = kernel(xx).evaluate()
    L_ref = torch.linalg.cholesky(K_ref + 0.1 * torch.eye(Ns + Nt)).detach().numpy()
    cholesky_decomp._update_cholesky(xx[Ns:], 0.1)
    assert np.allclose(L_ref, cholesky_decomp._decomposer._L.detach().numpy(), atol=1e-5)
    cholesky_decomp = SeparableCholeskyDecomposition(kernel)
    cholesky_decomp.new_cholesky(xx[:Ns], 0.1)
    assert np.allclose(L_ref, cholesky_decomp.new_cholesky(xx[Ns:], 0.1).detach().numpy(), atol=1e-5)


if __name__ == "__main__":
    test_separable_cholesky_tensorflow()
    test_separable_cholesky_torch()
