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

from alef.bayesian_optimization.base_candidate_generator import CandidateGenerator
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EvolutionaryOptimizerObjects:
    """
    Simple evolutionary optimizer over structured objects. It starts with an initial population. In the selection
    step it selects the self.survival_rate*100 % best candidates. In the reproduction step the selected candidates generate
    offspring via calling get_around_candidate_for_evolutionary_opt method in the candidate_generator object. The new generation
    is formed via the offspring and the survivors.
    """

    def __init__(self, population_size, num_offspring, candidate_generator: CandidateGenerator) -> None:
        self.candidate_generator = candidate_generator
        self.population_size = population_size
        self.num_offspring = num_offspring
        self.survival_rate = 1 / (self.num_offspring + 1)
        self.current_maximizer = None
        self.current_maximizer_value = -1 * np.infty

    def maximize(self, func, steps):
        """
        Maximize the objective function.
        """
        logger.info("EA - Start optimization")
        population = self.candidate_generator.get_initial_for_evolutionary_opt(n_initial=self.population_size)
        for step in range(0, steps):
            print("Step " + str(step + 1) + "/" + str(steps))
            if step > 0:
                survivors = self.select(population, function_values)  # noqa: F821
                population = self.reproduce(survivors)
            function_values = func(population)
            fittest_index = np.argmax(function_values)
            fittest_individual = population[fittest_index]
            fitness_fittest_indiviudal = function_values[fittest_index]
            logger.info("Current maximizer: ")
            logger.info(fittest_individual)
            logger.info("Function value maximizer:")
            logger.info(fitness_fittest_indiviudal)
            if self.current_maximizer_value < fitness_fittest_indiviudal:
                self.current_maximizer_value = fitness_fittest_indiviudal
                self.current_maximizer = fittest_individual

        logger.info("EA - Optimization complete")
        return self.current_maximizer, self.current_maximizer_value

    def select(self, population, function_values):
        """
        Select the best candidates from the population.
        """
        sorted_indexes = np.argsort(function_values)
        n = sorted_indexes.shape[0]
        num_survive = int(self.population_size * self.survival_rate)
        survivor_indexes = sorted_indexes[n - num_survive : n]
        survivors = []
        for index in survivor_indexes:
            survivors.append(population[index])
        return survivors

    def reproduce(self, survivors):
        """
        Reproduce new candidates from the survivors.
        """
        new_generation = []
        for survivor in survivors:
            new_generation += self.candidate_generator.get_around_candidate_for_evolutionary_opt(
                survivor, self.num_offspring
            )
        return new_generation + survivors
