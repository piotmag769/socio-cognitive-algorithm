import random
from typing import List

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.algorithm.singleobjective.genetic_algorithm import S


class RepelWorstGravityMultistep(GeneticAlgorithm):
    N: int = 5

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception("Wrong number of parents")

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)

                # "Repeal Worst Gravity" mutation
                if len(offspring_population) > 0:
                    # get worst individuals in current population
                    worst_individuals = self.solutions[-RepelWorstGravityMultistep.N :]
                    # use all worst individuals as repellers
                    # with given probability make current offspring's genes a negation of repeller's
                    for repeller in worst_individuals:
                        for l in range(len(solution.variables[0])):
                            rand = random.random()
                            if rand <= self.mutation_operator.probability:
                                solution.variables[0][l] = not repeller.variables[0][l]

                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def get_name(self) -> str:
        return "GA with 'Repel Worst Gravity Multistep' mutation"
