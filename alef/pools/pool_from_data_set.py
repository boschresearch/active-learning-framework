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

from typing import Union, List
from alef.pools.pool_from_data import PoolFromData
from alef.data_sets.base_data_set import StandardDataSet


class PoolFromDataSet(PoolFromData):
    def __init__(
        self,
        dataset: StandardDataSet,
        input_idx: List[Union[int, bool]],
        output_idx: List[Union[int, bool]] = [0],
        # observation_noise: float = None,
        # seed:int=123,
        # set_seed:bool=False
    ):
        x, y = dataset.get_complete_dataset()
        super().__init__(
            x[..., input_idx],
            y[..., output_idx],
            data_is_noisy=True,
            # observation_noise=observation_noise,
            # seed=seed,
            # set_seed=set_seed
        )


if __name__ == "__main__":
    from alef.data_sets.pytest_set import PytestSet

    dataset = PytestSet()
    dataset.load_data_set()

    pool = PoolFromDataSet(dataset, [0, 1, 2], [0])

    xx, yy = pool.get_full_data()
    print(xx.shape)
    print(yy.shape)
