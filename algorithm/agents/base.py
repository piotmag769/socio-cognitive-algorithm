from functools import cmp_to_key
from math import ceil
from typing import Optional

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.solution import Solution


class BaseAgent:
    POPULATION_PART_TO_SWAP = 0.1
    last_shared_solutions = None
    id = (-1,)

    def __init__(self, algorithm: GeneticAlgorithm, *args, **kwargs):
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

    def remove_solutions(self, solutions):
        self.algorithm.solutions = [
            solution
            for solution in self.algorithm.solutions
            if solution not in solutions
        ]

    def __eq__(self, other) -> bool:
        # Two different objects are always unequal.
        return id(self) == id(other)

    def __hash__(self) -> int:
        return hash(id(self))

    @classmethod
    def name(cls) -> str:
        return cls.__name__
