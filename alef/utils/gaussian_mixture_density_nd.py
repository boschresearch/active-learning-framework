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


from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt


class GaussianMixtureDensityNd:
    def __init__(self, weights, mus, covs):
        self.weights = weights
        self.mus = mus
        self.covs = covs
        assert self.weights.shape[0] == self.mus.shape[0]
        assert self.weights.shape[0] == self.covs.shape[0]

    def p(self, x):
        p = np.zeros((x.shape[:-1]))

        for i, weight in enumerate(self.weights):
            p += weight * multivariate_normal.pdf(x, self.mus[i], self.covs[i])
        return p

    def plot(self, a, b, step):
        x, y = np.mgrid[a:b:step, a:b:step]
        position = np.dstack((x, y))
        print(position.shape)
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.contourf(x, y, self.p(position), levels=40)
        plt.show()

    def mean(self):
        mean = np.average(self.mus, axis=0, weights=self.weights)
        return mean

    def covariance_matrix(self):
        covariance = np.average(self.covs, axis=0, weights=self.weights)
        diff_means = self.mus - self.mean()
        if len(diff_means.shape) == 1:
            diff_means = np.expand_dims(diff_means, axis=1)
        counter = 0
        for diff in diff_means:
            weight = self.weights[counter]
            diff_expanded = np.expand_dims(diff, axis=1)
            c_diff = np.matmul(diff_expanded, diff_expanded.T)
            covariance += weight * c_diff
        return covariance

    def entropy_gaussian_approx(self):
        mu = self.mean()
        cov = self.covariance_matrix()
        mvn = multivariate_normal(mu, cov)
        return mvn.entropy()

    def plot_gaussian_approx(self, a, b, step):
        mu = self.mean()
        cov = self.covariance_matrix()
        mvn = multivariate_normal(mu, cov)
        x, y = np.mgrid[a:b:step, a:b:step]
        position = np.dstack((x, y))
        print(position.shape)
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.contourf(x, y, mvn.pdf(position), levels=40)
        plt.show()


if __name__ == "__main__":
    weights = np.array([0.4, 0.6])
    means = np.array([np.array([2, 1]), np.array([2, 1.6])])
    covs = np.array([np.array([[0.1, -0.05], [-0.05, 0.1]]), np.array([[0.1, 0], [0, 0.1]])])
    gmm = GaussianMixtureDensityNd(weights, means, covs)
    gmm.p(np.array([[2, 2], [1, 1.5], [2, 3]]))
    gmm.plot(0.0, 3.0, 0.01)
    gmm.plot_gaussian_approx(0.0, 3.0, 0.01)
    print(gmm.entropy_gaussian_approx())
    gmm.covariance_matrix()
    gmm.mean()
