import random
import numpy as np

from typing import List
from math import ceil

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.algorithm.singleobjective.genetic_algorithm import S


class FollowBestDistinctGA(GeneticAlgorithm):
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

                # "Follow Best Distinct" mutation
                if len(offspring_population) > 0:
                    # get top individuals in current population
                    best_individuals = self.solutions[: FollowBestDistinctGA.N]

                    # calculate std deviations for top individuals for each gene position
                    number_of_genes = len(self.solutions[0].variables[0])
                    std_devs = []
                    for bit_position in range(number_of_genes):
                        std_devs.append(
                            np.std(
                                [
                                    solution.variables[0][bit_position]
                                    for solution in best_individuals
                                ],
                                ddof=1,
                            )
                        )
                    # calculate probabilities from deviations via softmax
                    probabilities = softmax(std_devs)

                    # choose k genes positions based on probabilities
                    K = ceil(
                        solution.number_of_variables
                        * self.mutation_operator.probability
                    )
                    positons_of_genes_to_mutate = random.choices(
                        list(range(number_of_genes)), weights=probabilities, k=K
                    )

                    # randomly select one to be a teacher
                    teacher = best_individuals[
                        random.randint(0, len(best_individuals) - 1)
                    ]
                    # assign chosen teacher's genes to current offspring's
                    for k in range(solution.number_of_variables):
                        for l in positons_of_genes_to_mutate:
                            solution.variables[k][l] = teacher.variables[k][l]

                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def get_name(self) -> str:
        return "GA with 'Follow Best Distinct' mutation"


def softmax(array: list[float]):
    np_array = np.array(array)
    exp_array = np.exp(np_array)
    return list(exp_array / np.sum(exp_array))
