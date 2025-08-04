import random
from typing import List

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.algorithm.singleobjective.genetic_algorithm import S


class FollowBestGA(GeneticAlgorithm):
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

                # "Follow Best" mutation
                if len(offspring_population) > 0:
                    # get top individuals in current population
                    best_individuals = self.solutions[: FollowBestGA.N]
                    # randomly select one to be a teacher
                    teacher = best_individuals[
                        random.randint(0, len(best_individuals) - 1)
                    ]
                    # with given probability assign teacher's genes to current offspring's
                    for l in range(len(solution.variables[0])):
                        rand = random.random()
                        if rand <= self.mutation_operator.probability:
                            solution.variables[0][l] = teacher.variables[0][l]

                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def get_name(self) -> str:
        return "GA with 'Follow Best' mutation"
