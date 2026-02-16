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
from typing import Union, Sequence
from alef.enums.data_structure_enums import OutputType
from alef.pools.pool_with_safety_from_oracle import PoolWithSafetyFromOracle
from alef.oracles import GPOracleFromData
from alef.data_sets.engine import Engine1, Engine2
from alef.models.model_factory import ModelFactory
from alef.configs.models.gp_model_for_engine1_config import (
    Engine1GPModelBEConfig,
    Engine1GPModelTExConfig,
    Engine1GPModelPI0vConfig,
    Engine1GPModelPI0sConfig,
    Engine1GPModelHCConfig,
    Engine1GPModelNOxConfig,
)
from alef.configs.models.gp_model_for_engine2_config import (
    Engine2GPModelBEConfig,
    Engine2GPModelTExConfig,
    Engine2GPModelPI0vConfig,
    Engine2GPModelPI0sConfig,
    Engine2GPModelHCConfig,
    Engine2GPModelNOxConfig,
)
from alef.configs.models.mogp_model_so_for_engine_config import (
    EngineMOGPModelBEConfig,
    EngineMOGPModelTExConfig,
    EngineMOGPModelPI0vConfig,
    EngineMOGPModelPI0sConfig,
    EngineMOGPModelHCConfig,
    EngineMOGPModelNOxConfig,
)
import os


_path = ""


class EnginePool(PoolWithSafetyFromOracle):
    def __init__(
        self,
        data_folder: str = _path,
        engine1or2: int = 1,
        input_idx: Sequence[Union[int, bool]] = [0, 1, 2, 3],
        output_idx: Sequence[Union[int, bool]] = [0],
        safety_idx: Sequence[Union[int, bool]] = [1, 3, 4, 5, 6],
        constrain_input: bool = False,
        flip_output: bool = False,
        context_idx: Sequence[int] = [],
        context_values: Sequence[float] = [],
        seed: int = 123,
        set_seed: bool = False,
    ):
        if engine1or2 == 1:
            engine_class = Engine1
        elif engine1or2 == 2:
            engine_class = Engine2
        else:
            assert False
        data_set = engine_class(data_folder)
        data_set.constrain_input = constrain_input
        data_set.load_data_set()

        X_raw, Y_raw = data_set.get_complete_dataset()
        X = X_raw[..., input_idx]
        Y = Y_raw[..., output_idx]
        Z = Y_raw[..., safety_idx]

        np.append(output_idx, safety_idx)

        Y_name = list(np.array(data_set.output_names)[output_idx])
        Z_name = list(np.array(data_set.output_names)[safety_idx])

        assert len(Y_name) == 1
        if flip_output:
            oracle = GPOracleFromData(self.__name2model(Y_name[0], engine1or2), X, -Y)
        else:
            oracle = GPOracleFromData(self.__name2model(Y_name[0], engine1or2), X, Y)
        safety_oracles = []
        for i in range(len(Z_name)):
            so = GPOracleFromData(self.__name2model(Z_name[i], engine1or2), X, Z[:, i, None])
            safety_oracles.append(so)

        super().__init__(oracle, safety_oracles, seed, set_seed)
        self.__use_context = False
        self.set_context(context_idx, context_values)

    def set_context(self, context_idx: Sequence[int] = [], context_values: Sequence[float] = []):
        if len(context_idx) > 0:
            self.oracle.set_context(context_idx, context_values)
            for so in self.safety_oracle:
                so.set_context(context_idx, context_values)
            self.__use_context = True

    def get_context_status(self, *args, **kwargs):
        return self.oracle.get_context_status(*args, **kwargs)

    def get_grid_data(self, n_per_dim: int, noisy: bool = True):
        """
        input: n_per_dim noisy
        """
        if not hasattr(self.oracle, "get_grid_data"):
            raise NotImplementedError(f"{self.oracle.__class__.__name__} does not have 'get_grid_data' method")
        if np.any([not hasattr(so, "get_grid_data") for so in self.safety_oracle]):
            raise NotImplementedError("At least one safety_oracle does not have 'get_grid_data' method")

        X, Y = self.oracle.get_grid_data(n_per_dim, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for j in range(len(self.safety_oracle)):
            Z[:, j] = self.safety_oracle[j].query_multiple_points_in_sequence(X, noisy).reshape(-1)
        return X, Y, Z

    def load_grid_data(self, folder: str):
        X_path = os.path.join(folder, "X_grid.txt")
        Y_path = os.path.join(folder, "Y_grid.txt")
        Z_path = os.path.join(folder, "Z_grid.txt")
        return self.load_data(X_path, Y_path, Z_path)

    def get_random_data(self, n, noisy=True):
        X, Y = self.oracle.get_random_data(n, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for j in range(len(self.safety_oracle)):
            Z[:, j] = self.safety_oracle[j].query_multiple_points_in_sequence(X, noisy).reshape(-1)
        return X, Y, Z

    def get_random_data_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]], noisy: bool = True
    ):
        X, Y = self.oracle.get_random_data_in_box(n, a, box_width, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for j in range(len(self.safety_oracle)):
            Z[:, j] = self.safety_oracle[j].query_multiple_points_in_sequence(X, noisy).reshape(-1)
        return X, Y, Z

    def discretize_random(self, n: int):
        r"""
        set x randomly from the space defined in the oracle (get discretized input space from the oracle)
        """
        a, b = self.oracle.get_contextual_box_bound()
        x = np.random.uniform(a, b, size=(n, self.get_dimension()))
        self.set_data(x)

    def discretize_random_in_box(
        self, n: int, a: Union[float, Sequence[float]], box_width: Union[float, Sequence[float]]
    ):
        r"""
        set x randomly from the space defined in the oracle (get discretized input space from the oracle)
        """
        x = self.oracle._get_random_x_in_box(n, a, box_width)
        self.set_data(x)

    def get_variable_dimension(self):
        return self.oracle.get_variable_dimension()

    def __name2model(self, name: str, engine1or2: int):
        if engine1or2 == 1:
            if name == "be":
                return ModelFactory.build(Engine1GPModelBEConfig())
            elif name == "T_Ex":
                return ModelFactory.build(Engine1GPModelTExConfig())
            elif name == "PI0v":
                return ModelFactory.build(Engine1GPModelPI0vConfig())
            elif name == "PI0s":
                return ModelFactory.build(Engine1GPModelPI0sConfig())
            elif name == "HC":
                return ModelFactory.build(Engine1GPModelHCConfig())
            elif name == "NOx":
                return ModelFactory.build(Engine1GPModelNOxConfig())
            else:
                raise NotImplementedError("invalid index for engine datasets")
        elif engine1or2 == 2:
            if name == "be":
                return ModelFactory.build(Engine2GPModelBEConfig())
            elif name == "T_Ex":
                return ModelFactory.build(Engine2GPModelTExConfig())
            elif name == "PI0v":
                return ModelFactory.build(Engine2GPModelPI0vConfig())
            elif name == "PI0s":
                return ModelFactory.build(Engine2GPModelPI0sConfig())
            elif name == "HC":
                return ModelFactory.build(Engine2GPModelHCConfig())
            elif name == "NOx":
                return ModelFactory.build(Engine2GPModelNOxConfig())
            else:
                raise NotImplementedError("invalid index for engine datasets")


class EngineCorrelatedPool(EnginePool):
    def __init__(
        self,
        data_folder: str = _path,
        engine1or2: int = 1,
        input_idx: Sequence[Union[int, bool]] = [0, 1, 2, 3],
        output_idx: Sequence[Union[int, bool]] = [0],
        safety_idx: Sequence[Union[int, bool]] = [1, 3, 4, 5, 6],
        constrain_input: bool = False,
        flip_output: bool = False,
        context_idx: Sequence[int] = [],
        context_values: Sequence[float] = [],
        seed: int = 123,
        set_seed: bool = False,
    ):
        super().__init__(
            data_folder=data_folder,  # not needed
            engine1or2=1,  # not needed
            input_idx=input_idx,  # not needed
            output_idx=[0],  # not needed
            safety_idx=[0],  # not needed
            constrain_input=True,  # not needed
            flip_output=False,  # not needed
            context_idx=context_idx,
            context_values=context_values,
            seed=seed,
            set_seed=set_seed,
        )

        data_set = Engine1(data_folder)
        data_set.constrain_input = constrain_input
        data_set.load_data_set()

        X_raw, Y_raw = data_set.get_complete_dataset()
        X_s = np.hstack((X_raw[..., input_idx], np.zeros([X_raw.shape[0], 1])))
        Y_s = Y_raw[..., output_idx]
        Z_s = Y_raw[..., safety_idx]

        data_set = Engine2(data_folder)
        data_set.constrain_input = constrain_input
        data_set.load_data_set()

        X_raw, Y_raw = data_set.get_complete_dataset()
        X_t = np.hstack((X_raw[..., input_idx], np.ones([X_raw.shape[0], 1])))
        Y_t = Y_raw[..., output_idx]
        Z_t = Y_raw[..., safety_idx]

        X = np.vstack((X_s, X_t))
        Y = np.vstack((Y_s, Y_t))
        Z = np.vstack((Z_s, Z_t))

        Y_name = list(np.array(data_set.output_names)[output_idx])
        Z_name = list(np.array(data_set.output_names)[safety_idx])

        assert len(Y_name) == 1

        if flip_output:
            oracle = GPOracleFromData(self.__name2model(Y_name[0]), X, -Y, OutputType.MULTI_OUTPUT_FLATTENED)
        else:
            oracle = GPOracleFromData(self.__name2model(Y_name[0]), X, Y, OutputType.MULTI_OUTPUT_FLATTENED)
        oracle.set_output_idx(engine1or2 - 1)

        safety_oracles = []
        for i in range(len(Z_name)):
            so = GPOracleFromData(self.__name2model(Z_name[i]), X, Z[:, i, None], OutputType.MULTI_OUTPUT_FLATTENED)
            so.set_output_idx(engine1or2 - 1)
            safety_oracles.append(so)

        self.oracle = oracle
        self.safety_oracle = safety_oracles
        self.set_context(context_idx, context_values)

    def __name2model(self, name: str):
        if name == "be":
            return ModelFactory.build(EngineMOGPModelBEConfig())
        elif name == "T_Ex":
            return ModelFactory.build(EngineMOGPModelTExConfig())
        elif name == "PI0v":
            return ModelFactory.build(EngineMOGPModelPI0vConfig())
        elif name == "PI0s":
            return ModelFactory.build(EngineMOGPModelPI0sConfig())
        elif name == "HC":
            return ModelFactory.build(EngineMOGPModelHCConfig())
        elif name == "NOx":
            return ModelFactory.build(EngineMOGPModelNOxConfig())
        else:
            raise NotImplementedError("invalid index for engine datasets")


if __name__ == "__main__":
    pool = EnginePool(engine1or2=2, context_idx=[2, 3], context_values=[1.29, 0.5447])
    pool.discretize_random(2000)
    print(np.where(pool.possible_queries()[:, 2] != 1.29)[0])
    print(np.where(pool.possible_queries()[:, 3] != 0.5447)[0])
    X, Y, Z = pool.get_grid_data(30)
    print(np.where(X[:, 2] != 1.29)[0])
    print(np.where(X[:, 3] != 0.5447)[0])
    X, Y, Z = pool.get_random_data(1000)
    print(np.where(X[:, 3] != 0.5447)[0])
    print(pool.get_box_bounds())
    X, Y, Z = pool.get_random_data_in_box(50, [-1.2, 0.37, 1.29, 0.5], [0.5, 1.5, 1, 1], True)
    print(np.where(X[:, 2] != 1.29)[0])
    print(np.where(X[:, 3] != 0.5447)[0])

    X, Y, Z = pool.get_random_constrained_data(
        300, True, [-np.inf, -np.inf, -0.25, -np.inf, -np.inf], [1.6, np.inf, np.inf, -0.2, np.inf]
    )
    print(np.where(X[:, 2] != 1.29)[0])
    print(np.where(X[:, 3] != 0.5447)[0])

    X, Y, Z = pool.get_random_constrained_data_in_box(
        300,
        [-1.2, 0.37, 1.29, 0.5],
        [0.5, 1.5, 1, 1],
        True,
        [-np.inf, -np.inf, -0.25, -np.inf, -np.inf],
        [1.6, np.inf, np.inf, -0.2, np.inf],
    )
    print(np.where(X[:, 2] != 1.29)[0])
    print(np.where(X[:, 3] != 0.5447)[0])
