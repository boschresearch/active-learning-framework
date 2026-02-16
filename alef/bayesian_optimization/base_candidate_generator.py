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

from abc import ABC, abstractmethod
from typing import List


class CandidateGenerator(ABC):
    """
    Generator of objects for Bayesian Optimization over objects. As it is not clear how objects are generated (in comparision to real numbers that can be generated from a grid) to optimize
    the acquisition function we need this class that provides an interface and absorbes all the details of generating a particular kind of objects. It provides methods
    for the different kinds of optimizing the acquistion fucntion [TRAILING,EVOLUTIONARY]. Trailing optimization means that BO keeps an active set of objects that is updated in
    each iteration - objects with low acquisition value are deleted and new objects around the current best object are added to the active set.
    """

    @abstractmethod
    def get_initial_candidates_trailing(self) -> List[object]:
        """
        Initial candidate objects for the trailing optimization.
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_canditates(self, n_candidates: int, seed=100, set_seed=False) -> List[object]:
        """
        Retrieve n_candidates random candidates.
        """
        raise NotImplementedError

    @abstractmethod
    def get_additional_candidates_trailing(self, best_current_candidate: object) -> List[object]:
        """
        Retrieve additional candidates for the active set.
        """
        raise NotImplementedError

    @abstractmethod
    def get_initial_for_evolutionary_opt(self, n_initial):
        """
        Retrieve initial population for EvolutionaryOptimizerObjects.
        """
        raise NotImplementedError

    @abstractmethod
    def get_around_candidate_for_evolutionary_opt(self, candidate: object, n_around_candidate: int):
        """
        Retrieve candidate objects that are close to the input candidate.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset_recursivly_generated(self, n_data: int, n_per_step: int) -> List[object]:
        """
        Retrieve a dataset generated recursively.
        """
        raise NotImplementedError
