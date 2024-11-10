from math import ceil
from functools import cmp_to_key

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.solution import Solution

from .base import BaseAgent


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
