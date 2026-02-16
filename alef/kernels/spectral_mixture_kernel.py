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

# Source: https://github.com/imsrgadich/gprsm/blob/master/gprsm/spectralmixture.py

import gpflow
import tensorflow as tf
import numpy as np
from gpflow.utilities import positive
from gpflow.kernels import Kernel


from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from alef.kernels.scale_interface import StationaryKernelGPflow

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float

gpflow.config.set_default_jitter(1e-4)


class SpectralMixtureKernel(Kernel, InputInitializedKernelInterface, StationaryKernelGPflow):
    def __init__(self, num_mixtures: int, **kwargs):
        """
        - num_mixtures is the number of mixtures; denoted as Q in
        Wilson 2013.
        - mixture_weights
        - mixture_variance is
        - mixture_means is the list (or array) of means of the
        mixtures.
        - input_dim is the dimension of the input to the kernel.
        - active_dims is the dimension of the X which needs to be used.
        References:
        http://hips.seas.harvard.edu/files/wilson-extrapolation-icml-2013_0.pdf
        http://www.cs.cmu.edu/~andrewgw/typo.pdf
        """
        super().__init__()
        # Q(num_of_mixtures)=1 then SM kernel is SE Kernel.
        if num_mixtures == 1:
            print("Using default mixture = 1")

        # number of mixtures is non trainable.
        self.num_mixtures = num_mixtures
        self.mixture_weights = None
        self.mixture_scales = None
        self.mixture_means = None

    def K(self, X1, X2=None):
        if self.mixture_weights == None or self.mixture_means == None or self.mixture_scales == None:
            raise RuntimeError(
                "Parameters of spectral mixture kernel not initialized.\
                                    Run `sm_kern_object.initialize_(train_x,train_y)`."
            )
            # initialization can only be done by user as it needs target data as well.
        if X2 is None:
            X2 = X1

        # get absolute distances
        X1 = tf.transpose(tf.expand_dims(X1, -1), perm=[1, 0, 2])  # D x N1 x 1
        X2 = tf.expand_dims(tf.transpose(X2, perm=[1, 0]), -2)  # D x 1 x N2

        # r = tf.abs(tf.subtract(X1, X2))
        r = tf.subtract(X1, X2)  # D x N1 x N2

        cos_term = tf.multiply(tf.tensordot(self.mixture_means, r, axes=((1), (0))), 2.0 * np.pi)
        # num_mixtures x N1 x N2

        scales_expand = tf.expand_dims(tf.expand_dims(self.mixture_scales, -2), -2)
        # D x 1 x 1 x num_mixtures
        r_tile = tf.tile(tf.expand_dims(r, -1), (1, 1, 1, self.num_mixtures))
        # D x N1 x N2 x num_mixtures
        exp_term = tf.multiply(
            tf.transpose(tf.reduce_sum(tf.square(tf.multiply(r_tile, scales_expand)), 0), perm=[2, 0, 1]),
            -2.0 * np.pi**2,
        )
        # num_mixtures x N1 x N2

        weights = tf.expand_dims(tf.expand_dims(self.mixture_weights, -1), -1)
        weights = tf.tile(weights, (1, tf.shape(X1)[1], tf.shape(X2)[2]))
        return tf.reduce_sum(tf.multiply(weights, tf.multiply(tf.exp(exp_term), tf.cos(cos_term))), 0)

    def K_diag(self, X):
        # just the sum of weights. Weights represent the signal
        # variance.
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.reduce_sum(self.mixture_weights, 0))

    def initialize_parameters(self, x_data, y_data):
        mixture_weights, mixture_means, mixture_scales = sm_init(x_data, y_data, self.num_mixtures)
        self.set_parameters(mixture_weights, mixture_means, mixture_scales)

    def set_parameters(self, mixture_weights, mixture_means, mixture_scales):
        self.mixture_weights = gpflow.Parameter(f64(mixture_weights), transform=positive())
        self.mixture_scales = gpflow.Parameter(f64(mixture_scales), transform=positive())
        self.mixture_means = gpflow.Parameter(f64(mixture_means), transform=positive())


def sm_init(train_x, train_y, num_mixtures):
    """
    For initialization of the parameters for the Spectral Mixture
    Kernel.
    :param train_x: input data
    :param train_y: target data
    :param num_mixtures: number of mixtures
    :return: param_name       dimensions
             ----------       ----------
             mixture weights| num_mixtures x 1
             mixture means  | num_mixtures x input_dim
             mixture scales | input_dim x num_mixtures
    """
    assert isinstance(num_mixtures, int)
    assert train_x.shape[0] == train_y.shape[0]

    input_dim = np.shape(train_x)[1]  # type: int

    if np.size(train_x.shape) == 1:
        train_x = np.expand_dims(train_x, -1)

    if np.size(train_x.shape) == 2:
        train_x = np.expand_dims(train_x, 0)

    train_x_sort = np.copy(train_x)
    train_x_sort.sort(axis=1)

    max_dist = np.squeeze(train_x_sort[:, -1, :] - train_x_sort[:, 0, :])
    print(max_dist)

    min_dist_sort = np.squeeze(np.abs(train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]))
    min_dist = np.zeros([input_dim], dtype=float)

    # min of each data column could be zero. Hence, picking minimum which is not zero
    for ind in np.arange(input_dim):
        try:
            min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort[:, ind] > 0), axis=1), ind]
        except:
            min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort > 0), axis=1)]

    print(min_dist)
    # for random restarts during batch processing. We need to initialize at every
    # batch. Lock the seed here.
    seed = np.random.randint(low=1, high=2**31)
    np.random.seed(seed)

    # Inverse of lengthscales should be drawn from truncated Gaussian |N(0, max_dist^2)|
    # dim: Q x D
    # mixture_scales = tf.multiply(,tf.cast(max_dist,dtype=tf.float32)**(-1)

    mixture_scales = (
        np.multiply(np.abs(np.random.randn(num_mixtures, input_dim)), np.expand_dims(max_dist, axis=0))
    ) ** (-1)
    # Draw means from Unif(0, 0.5 / minimum distance between two points), dim: Q x D
    # the nyquist is half of maximum frequency. TODO
    nyquist = np.divide(0.5, min_dist)
    mixture_means = np.multiply(np.random.rand(num_mixtures, input_dim), np.expand_dims(nyquist, 0))
    # mixture_means[0, :] = 0

    # Mixture weights should be roughly the std  of the y values divided by
    # the number of mixtures
    # dim: 1 x Q
    mixture_weights = np.divide(np.std(train_y, axis=0), num_mixtures) * np.ones(num_mixtures)

    return mixture_weights, mixture_means, mixture_scales.T
