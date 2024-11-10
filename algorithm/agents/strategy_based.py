import random

from enum import Enum
from functools import cmp_to_key
from math import ceil

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.solution import Solution

from .base import BaseAgent


class SendStrategy(Enum):
    Best = 1
    Average = 2
    Worst = 3
    Outlying = 4  # TODO
    Random = 5
    Dont = 6


class AcceptStrategy(Enum):
    Always = 1
    Better = 2
    Different = 3  # TODO
    Reject = 4


class StrategyAgent(BaseAgent):
    def __init__(
        self,
        algorithm: GeneticAlgorithm,
        send_strategy: SendStrategy,
        accept_strategy: AcceptStrategy,
    ):
        self.algorithm = algorithm
        self.send_strategy = send_strategy
        self.accept_strategy = accept_strategy

    def get_solutions_to_share(self, agent_to_share_with) -> list[Solution]:
        number_of_solutions = len(self.algorithm.solutions)
        solutions_to_share = ceil(
            number_of_solutions * BaseAgent.POPULATION_PART_TO_SWAP
        )

        if self.send_strategy is SendStrategy.Best:
            return self.algorithm.solutions[:solutions_to_share]
        elif self.send_strategy is SendStrategy.Average:
            return self.algorithm.solutions[
                (number_of_solutions - solutions_to_share)
                // 2 : (number_of_solutions + solutions_to_share)
                // 2
            ]
        elif self.send_strategy is SendStrategy.Worst:
            return self.algorithm.solutions[-solutions_to_share:]
        elif self.send_strategy is SendStrategy.Random:
            return random.sample(self.algorithm.solutions, solutions_to_share)
        elif self.send_strategy is SendStrategy.Dont:
            return []

    def use_shared_solutions(
        self,
        shared_solutions: list[Solution],
        agent_sharing_the_solution,
    ):
        if self.accept_strategy is AcceptStrategy.Always:
            self.algorithm.solutions = [shared_solutions + self.algorithm.solutions][
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
        elif self.accept_strategy is AcceptStrategy.Reject:
            pass
