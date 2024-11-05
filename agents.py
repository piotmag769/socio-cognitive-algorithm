from math import ceil
from functools import cmp_to_key
from enum import Enum
import random

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.solution import Solution


class BaseAgent:
    POPULATION_PART_TO_SWAP = 0.1

    def __init__(self, algorithm: GeneticAlgorithm):
        self.algorithm = algorithm

    def get_solutions_to_share(self, agent_to_share_with) -> list[Solution]:
        number_of_solutions = len(self.algorithm.solutions)
        solutions_to_share = ceil(
            number_of_solutions * BaseAgent.POPULATION_PART_TO_SWAP
        )

        return self.algorithm.solutions[0:solutions_to_share]

    def use_shared_solutions(
        self,
        shared_solutions: list[Solution],
        agent_sharing_the_solution,
    ):
        self.algorithm.solutions.extend(shared_solutions)
        # Compare solutions (we assume they were already evaluated).
        self.algorithm.solutions.sort(
            key=cmp_to_key(self.algorithm.solution_comparator.compare)
        )
        # Get best solutions.
        self.algorithm.solutions = self.algorithm.solutions[
            : self.algorithm.population_size
        ]

    def __eq__(self, other):
        # Two different objects are always unequal.
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))


class AgentWithTrust(BaseAgent):
    # Trust level equal to i means that i-th best solution will be shared - the lower, the better.
    MAX_TRUST_LEVEL = 0

    def __init__(self, algorithm: GeneticAlgorithm):
        super().__init__(algorithm)
        self.trust_level_map = {}

    def get_solutions_to_share(self, agent_to_share_with) -> list[Solution]:
        if agent_to_share_with not in self.trust_level_map:
            self.trust_level_map[agent_to_share_with] = AgentWithTrust.MAX_TRUST_LEVEL

        trust_lvl = self.trust_level_map[agent_to_share_with]
        number_of_solutions = len(self.algorithm.solutions)
        solutions_to_share = ceil(
            number_of_solutions * AgentWithTrust.POPULATION_PART_TO_SWAP
        )

        index_of_best_solution_to_share = (
            trust_lvl
            if trust_lvl <= number_of_solutions - solutions_to_share
            else number_of_solutions - solutions_to_share
        )

        return self.algorithm.solutions[
            index_of_best_solution_to_share : (
                index_of_best_solution_to_share + solutions_to_share
            )
        ]

    def use_shared_solutions(
        self, shared_solutions: list[Solution], agent_sharing_the_solution
    ):
        self.algorithm.solutions.extend(shared_solutions)
        # Compare solutions (we assume they were already evaluated).
        self.algorithm.solutions.sort(
            key=cmp_to_key(self.algorithm.solution_comparator.compare)
        )
        # Get best solutions.
        self.algorithm.solutions = self.algorithm.solutions[
            : self.algorithm.population_size
        ]

        if agent_sharing_the_solution not in self.trust_level_map:
            self.trust_level_map[agent_sharing_the_solution] = (
                AgentWithTrust.MAX_TRUST_LEVEL
            )

        trust_change = 0
        for shared_solution in shared_solutions:
            if shared_solution in self.algorithm.solutions:
                # A useful solution was shared.
                trust_change -= 1
            else:
                # A useless solution was shared.
                trust_change += 1

        self.trust_level_map[agent_sharing_the_solution] = max(
            AgentWithTrust.MAX_TRUST_LEVEL,
            self.trust_level_map[agent_sharing_the_solution] + trust_change,
        )


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
