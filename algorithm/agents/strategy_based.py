from copy import deepcopy
import random

from enum import Enum
from functools import cmp_to_key
from math import ceil
from typing import Optional

import numpy as np

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.solution import Solution
from jmetal.core.problem import BinaryProblem

from .base import BaseAgent


class SendStrategy(Enum):
    Best = 1
    Average = 2
    Worst = 3
    Outlying = 4
    Random = 5
    Dont = 6


class AcceptStrategy(Enum):
    Always = 1
    Better = 2
    Different = 3
    Reject = 4


# 1. Modyfikacja zaufania na podstawie liczby przyjętych osobników (DONE)
# 2. Modyfikacja zaufania na podstawie czy najlepszy osobnik na wyspie ma lepszy fitness po akceptacji/po 10 iteracjach


# max_trust = 0, min zaufanie = -inf
# wysyłamy "najlepszych" osobników, ale bez trust(A, B) "najlepszych"
# dictionary: zaufanie - lokalne, reputacja - globalne
class TrustMechanism(Enum):
    Local = 1
    Global = 2

    def __str__(self):
        return self.name


class StrategyAgent(BaseAgent):
    # Trust level equal to i means that i-th best solution will be shared - the lower, the better.
    MAX_TRUST_LEVEL = 0

    def __init__(
        self,
        algorithm: GeneticAlgorithm,
        send_strategy: SendStrategy,
        accept_strategy: AcceptStrategy,
        trust_mechanism: Optional[TrustMechanism],
        trust: dict[BaseAgent, int],
        starting_trust: Optional[int] = MAX_TRUST_LEVEL,
        no_send_penalty: Optional[int] = 0,
        part_to_swap: Optional[float] = 0.1,
        id: int = -1,
    ):
        self.algorithm = algorithm
        self.send_strategy = send_strategy
        self.accept_strategy = accept_strategy
        self.starting_trust = starting_trust
        self.no_send_penalty = no_send_penalty
        self.part_to_swap = part_to_swap
        self.id = id

        if trust_mechanism is None:
            self.trust = None
        elif trust_mechanism is TrustMechanism.Global:
            self.trust = trust
        elif trust_mechanism is TrustMechanism.Local:
            self.trust = deepcopy(trust)
        else:
            assert False, "Unhandled case"

    def get_solutions_to_share(self, agent_to_share_with) -> list[Solution]:
        number_of_solutions = len(self.algorithm.solutions)
        solutions_to_share = ceil(number_of_solutions * self.part_to_swap)

        if self.trust is not None:
            if agent_to_share_with not in self.trust:
                self.trust[agent_to_share_with] = self.starting_trust

            trust_lvl = self.trust[agent_to_share_with]

            index_of_best_solution_to_share = (
                trust_lvl
                if trust_lvl <= number_of_solutions - solutions_to_share
                else number_of_solutions - solutions_to_share
            )
        else:
            index_of_best_solution_to_share = 0

        if self.send_strategy is SendStrategy.Best:
            chosen_solutions = self.algorithm.solutions[
                index_of_best_solution_to_share : (
                    index_of_best_solution_to_share + solutions_to_share
                )
            ]
        elif self.send_strategy is SendStrategy.Average:
            chosen_solutions = self.algorithm.solutions[
                (number_of_solutions - solutions_to_share)
                // 2 : (number_of_solutions + solutions_to_share)
                // 2
            ]
        elif self.send_strategy is SendStrategy.Worst:
            chosen_solutions = self.algorithm.solutions[
                (
                    -solutions_to_share - index_of_best_solution_to_share
                ) : -index_of_best_solution_to_share
            ]
        elif self.send_strategy is SendStrategy.Random:
            chosen_solutions = random.sample(
                self.algorithm.solutions, solutions_to_share
            )
        elif self.send_strategy is SendStrategy.Dont:
            chosen_solutions = []
        elif self.send_strategy is SendStrategy.Outlying:
            chosen_solutions = self.rank_outliers()[
                index_of_best_solution_to_share : (
                    solutions_to_share + index_of_best_solution_to_share
                )
            ]
        return chosen_solutions

    def use_shared_solutions(
        self,
        shared_solutions: list[Solution],
        agent_sharing_the_solution,
        starting_population_size,
    ):
        if self.accept_strategy is AcceptStrategy.Always:
            shared_solutions.extend(self.algorithm.solutions)
            self.algorithm.solutions = shared_solutions[
                : self.algorithm.population_size
            ]

            # Sort to keep in order.
            self.algorithm.solutions.sort(
                key=cmp_to_key(self.algorithm.solution_comparator.compare)
            )
        elif self.accept_strategy is AcceptStrategy.Better:
            self.algorithm.solutions.extend(shared_solutions)
            self.algorithm.solutions.sort(
                key=cmp_to_key(self.algorithm.solution_comparator.compare)
            )

            self.algorithm.solutions = self.algorithm.solutions[
                : self.algorithm.population_size
            ]
            if self.trust is not None:
                if agent_sharing_the_solution not in self.trust:
                    self.trust[agent_sharing_the_solution] = (
                        self.__class__.MAX_TRUST_LEVEL
                    )

                trust_change = 0
                if len(shared_solutions) > 0:
                    for shared_solution in shared_solutions:
                        if shared_solution in self.algorithm.solutions:
                            # A useful solution was shared.
                            trust_change -= 1
                        else:
                            # A useless solution was shared.
                            trust_change += 1
                else:
                    trust_change = self.no_send_penalty

                self.trust[agent_sharing_the_solution] = max(
                    self.__class__.MAX_TRUST_LEVEL,
                    self.trust[agent_sharing_the_solution] + trust_change,
                )

        elif self.accept_strategy is AcceptStrategy.Reject:
            pass
        elif self.accept_strategy is AcceptStrategy.Different:
            self.algorithm.solutions.extend(shared_solutions)
            self.algorithm.solutions = self.rank_outliers()[
                : self.algorithm.population_size
            ]

            self.algorithm.solutions.sort(
                key=cmp_to_key(self.algorithm.solution_comparator.compare)
            )

            if self.trust is not None:
                if agent_sharing_the_solution not in self.trust:
                    self.trust[agent_sharing_the_solution] = (
                        self.__class__.MAX_TRUST_LEVEL
                    )

                if len(shared_solutions) > 0:
                    trust_change = 0
                    for shared_solution in shared_solutions:
                        if shared_solution in self.algorithm.solutions:
                            # A useful solution was shared.
                            trust_change -= 1
                        else:
                            # A useless solution was shared.
                            trust_change += 1
                else:
                    trust_change = self.no_send_penalty

                self.trust[agent_sharing_the_solution] = max(
                    self.__class__.MAX_TRUST_LEVEL,
                    self.trust[agent_sharing_the_solution] + trust_change,
                )

        """
        In some cases (e.g. when Agent sends but does not accept solutions) the population size decreases.
        To keep the population size constant, we need to fill in the gaps.
        Currently we use method that creates a new solution by crossing existing solutions. 
        """
        if len(self.algorithm.solutions) < starting_population_size:
            parents_for_crossover = (
                self.algorithm.crossover_operator.get_number_of_parents()
            )
            mating_population = self.algorithm.solutions[
                : len(self.algorithm.solutions)
                - len(self.algorithm.solutions) % parents_for_crossover
            ]
            new_solutions = self.algorithm.reproduction(mating_population)
            self.algorithm.solutions.extend(
                new_solutions[
                    : starting_population_size - len(self.algorithm.solutions)
                ]
            )

        # Double Check
        assert (
            len(self.algorithm.solutions) == starting_population_size
        ), "Population refill is not enough!!!"

    # Returns solutions sorted by the dot product of its variables and the mean variables of all the solutions
    # in an ascending order.
    def rank_outliers(self):
        solutions = self.algorithm.solutions

        if isinstance(self.algorithm.problem, BinaryProblem):
            variables_mean = np.array(
                [solution.variables[0] for solution in solutions]
            ).mean(axis=0)
            ranked_sol = sorted(
                solutions,
                key=lambda solution: np.dot(solution.variables[0], variables_mean),
            )
        else:
            variables_mean = np.array(
                [solution.variables for solution in solutions]
            ).mean(axis=0)
            ranked_sol = sorted(
                solutions,
                key=lambda solution: np.dot(solution.variables, variables_mean),
            )
        return ranked_sol
