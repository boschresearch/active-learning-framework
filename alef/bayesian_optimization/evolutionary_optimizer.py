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


class EvolutionaryOptimizer:
    def __init__(self, population_size) -> None:
        self.population_size = population_size
        self.num_offspring = 5
        self.survival_rate = 1 / self.num_offspring
        self.current_maximizer = None
        self.current_maximizer_value = -1 * np.infty
        self.mutuation_strength = 0.1
        self.mutuation_strength_dicount = 0.8

    def maximize(self, func, dimension, bounds_a, bounds_b, steps):
        """
        Maximize the objective function.
        """
        print("EA - Start optimization")
        self.bounds_a = bounds_a
        self.bounds_b = bounds_b
        assert dimension == bounds_b.shape[0]
        population = np.random.uniform(low=bounds_a, high=bounds_b, size=(self.population_size, dimension))
        for step in range(0, steps):
            print("Step " + str(step + 1) + "/" + str(steps))
            if step > 0:
                survivors = self.select(population, function_values)  # noqa: F821
                population = self.reproduce(survivors)
            function_values = func(population)
            fittest_index = np.argmax(function_values)
            fittest_individual = population[fittest_index]
            fitness_fittest_indiviudal = function_values[fittest_index]
            if self.current_maximizer_value < fitness_fittest_indiviudal:
                self.current_maximizer_value = fitness_fittest_indiviudal
                self.current_maximizer = fittest_individual
            self.mutuation_strength = self.mutuation_strength * self.mutuation_strength_dicount
        print("EA - Optimization complete")
        return self.current_maximizer, self.current_maximizer_value

    def select(self, population, function_values):
        """
        Select the best candidates from the population.
        """
        sorted_indexes = np.argsort(function_values)
        sorted_population = population[sorted_indexes]
        num_survive = int(self.population_size * self.survival_rate)
        n = sorted_population.shape[0]
        survivors = sorted_population[n - num_survive : n]
        return survivors

    def reproduce(self, survivors):
        """
        Reproduce new candidates from the survivors.
        """
        new_generation = []
        for survivor in survivors:
            for j in range(0, self.num_offspring):
                child = self.mutate(survivor)
                new_generation.append(child)
        return np.array(new_generation)

    def mutate(self, survivor):
        """
        Mutate a survivor to create a new candidate.
        """
        bounds_diff = self.bounds_b - self.bounds_a
        delta = 0.5 * self.mutuation_strength * bounds_diff
        child_a = np.max(np.stack((survivor - delta, self.bounds_a)), axis=0)
        child_b = np.min(np.stack((survivor + delta, self.bounds_b)), axis=0)
        child = np.random.uniform(child_a, child_b)
        return child
