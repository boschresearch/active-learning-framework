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

import numpy as np
import tensorflow as tf
from gpflow.kernels import Static
from alef.utils.utils import tf_delta

"""
The following class is inspired from GPflow White kernel class
(https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/statics.py
Copyright 2017-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""


class Delta(Static):
    def K(self, X, X2=None) -> tf.Tensor:
        if X2 is None:
            d = tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
            return tf.linalg.diag(d)
        else:
            return self.variance * tf.cast(tf_delta(X, X2), self.variance.dtype)


class Delta_t(Static):
    def __init__(self, variance=1.0, reference_index=-1, active_dims=None):
        super().__init__(variance, active_dims)
        self.reference_index = reference_index

    def K_diag(self, X):
        ref_idx = tf.cast(tf.fill(tf.shape(X)[:-2], self.reference_index)[..., None, None], X.dtype)
        return self.variance * tf.cast(tf_delta(X, ref_idx)[..., 0], self.variance.dtype)

    def K(self, X, X2=None):
        if X2 is None:
            return tf.linalg.diag(self.K_diag(X))
        else:
            joint_idx = tf.cast(tf.matmul(X, X2, transpose_b=True), tf.int32)
            ref_idx = tf.cast(tf.square(self.reference_index), tf.int32)
            return tf.where(joint_idx == ref_idx, self.variance, tf.zeros(1, dtype=self.variance.dtype))


if __name__ == "__main__":
    k = Delta()

    x1 = np.reshape(np.arange(10) % 3, [-1, 1])
    x2 = np.reshape(np.arange(1, 11) % 4, [-1, 1])
    print(x1.reshape(-1))
    print(x2.reshape(-1))
    print(k(x1, x2))

    k = Delta_t(reference_index=1)
    print(k(x1))
    print(k(x1, x2))
