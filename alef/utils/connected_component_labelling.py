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

# from tssl.utils.disjoint_set_data_structure import DisjointSet
from disjoint_set import DisjointSet

"""
TwoPass algorithm is originally for pixel like data.
See
https://en.wikipedia.org/wiki/Connected-component_labeling
https://en.wikipedia.org/wiki/Hoshen%E2%80%93Kopelman_algorithm

I extend it to high dim data and to data where we have coordinate inputs
    (but still need to be perfectly distributed on grids).
"""


class TrivialDetector:
    """
    deal with 1D grid and safety masks
    """

    def cca(self, grid: np.ndarray, mask: np.ndarray):
        idx = self._grid_loc(grid)

        S = mask.reshape(-1)[idx]
        labels = np.zeros([len(S), 1], dtype=int)

        CurrentLabel = 1
        for n in range(len(labels)):
            if n == 0 and S[n] == 0:
                continue
            elif n == 0:
                labels[n] = CurrentLabel
            else:  # n >= 1
                if S[n] > 0:
                    labels[n] = CurrentLabel
                elif S[n - 1] > 0 and S[n] == 0:
                    CurrentLabel += 1

        labels[:] = labels[np.argsort(idx)]
        num_labels = int(max(labels))
        return num_labels, labels

    def _grid_loc(self, grid):
        S = grid.reshape(-1)
        return np.argsort(S)

    def label_points(self, points, reference_grid, reference_labels):
        r"""
        points: [N, 1] array, points to label
        reference_grid: [N_pool, 1] array, labeled grid as reference
        reference_labels: [N_pool,] or [N_pool, 1] array, labels of reference_grid

        return
        [N,] array, labels of points
        """
        x = np.atleast_2d(points)
        N = x.shape[0]
        labels = np.zeros(N, dtype=int)

        for ref_l in np.unique(reference_labels):
            if ref_l == 0:
                continue
            label_set = reference_grid[reference_labels.reshape(-1) == ref_l, :1]
            mask = np.logical_and(x[:, 0] <= max(label_set[:, 0]), x[:, 0] >= min(label_set[:, 0]))
            labels[mask] = ref_l

        return labels


class TwoPassHighDim:
    """
    deal with high dimensional grid and safety masks
    """

    def cca(self, grid: np.ndarray, mask: np.ndarray):
        n = grid.shape[0]
        idx, n_per_dim, dimension = self._grid_loc(grid)

        labels = np.zeros(n, dtype=int)
        # labelset = DisjointSet( np.arange(-1, n, dtype=int) ) # leave -1 for unoccupied relation
        labelset = DisjointSet({i: i for i in range(-1, n)})
        idx_map = {tuple(idx[i, :]): i for i in range(n)}
        occupied = mask.reshape([-1, 1]).astype(int)

        """
        First pass:
        if a grid is not occupied, relate the grid to background and go to the next grid
        otherwise
            relate all occupied neighbors to current grid
        """
        for i in range(n):
            if occupied[i] == 0:
                labelset.union(-1, i)
            else:
                neighbors = self._return_neighbor_locs(idx[i, :], n_per_dim)
                if neighbors.shape[0] == 0:
                    continue
                for nid in neighbors:
                    if occupied[idx_map[tuple(nid)]]:
                        labelset.union(i, idx_map[tuple(nid)])
        """
        Second pass:
        assign a label to each set of grids
        if a grid is occupied
            label the grid with the value corresponding to this set
        """
        # reprs = labelset.get_sets()
        # label_map = {p: i+1 for i, p in enumerate(reprs[reprs>=0])}
        # num_labels = len(reprs) - 1 # unoccupied points are all in set -1
        reprs = [i for i, _ in labelset.itersets(with_canonical_elements=True) if not labelset.connected(-1, i)]
        label_map = {p: i + 1 for i, p in enumerate(reprs)}
        num_labels = len(reprs)

        for i in range(n):
            if occupied[i] == 1:
                labels[i] = label_map[labelset.find(i)]

        return num_labels, labels.reshape([-1, 1])

    def _return_neighbor_locs(self, idx, n_per_dim):
        dim = len(n_per_dim.reshape(-1))
        coordinate = np.atleast_2d(idx).astype(int)
        assert coordinate.shape[0] == 1
        assert coordinate.shape[1] == dim

        neighbor_gap = -1 * np.eye(dim, dtype=int)
        # np.vstack((
        #    -1 * np.eye(dim, dtype=int),
        #    np.eye(dim, dtype=int)
        # ))
        output_locs = coordinate + neighbor_gap
        border_mask = np.all((output_locs >= [0] * dim) & (output_locs < n_per_dim.reshape(-1)), axis=1)
        return output_locs[border_mask]

    def _grid_loc(self, grid):
        dimension = grid.shape[1]
        grid_per_dim = {d: np.unique(grid[:, d]) for d in range(dimension)}
        n_per_dim = np.array([len(grid_per_dim[d]) for d in range(dimension)], dtype=int)

        idx = np.zeros_like(grid, dtype=int)

        for d in range(dimension):
            for i, xd in enumerate(grid_per_dim[d]):
                idx[grid[:, d] == xd, d] = i

        return idx, n_per_dim, dimension

    def label_points(self, points, reference_grid, reference_labels):
        r"""
        points: [N, D] array, points to label
        reference_grid: [N_pool, D] array, labeled grid as reference
        reference_labels: [N_pool,] or [N_pool, 1] array, labels of reference_grid

        return
        [N,] array, labels of points
        """
        x = np.atleast_2d(points)
        N, D = x.shape
        d_grid = np.zeros(D, dtype=float)

        for i in range(D):
            dx = np.unique(np.diff(reference_grid[:, i]))
            d_grid[i] = dx[dx > 0].min()

        labels = np.zeros(N, dtype=int)

        for ref_l in np.unique(reference_labels):
            if ref_l == 0:
                continue
            label_set = reference_grid[reference_labels.reshape(-1) == ref_l, :D]

            mask = np.any(
                np.all(
                    np.absolute(x[:, None, :D] - label_set[None, :, :D])
                    <= d_grid / 2,  # [N, N_pool, D], distance of x to reference_grid per dimension
                    axis=-1,
                ),  # [N, N_pool], is x in any cube centered at each reference_grid
                axis=-1,
            )
            labels[mask] = ref_l

        return labels
