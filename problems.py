# https://en.wikipedia.org/wiki/Test_functions_for_optimization
import numpy as np
import random

from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution


class LABS(BinaryProblem):
    def __init__(
        self,
        sequence_length: int = 10,
    ):
        super(LABS, self).__init__()
        self.number_of_bits = sequence_length

    def number_of_variables(self) -> int:
        return 1

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        solution.objectives[0] = energy_function(solution.variables[0])
        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(
            number_of_variables=self.number_of_variables(),
            number_of_objectives=self.number_of_objectives(),
        )

        new_solution.variables[0] = [
            random.randint(0, 1) == 0 for _ in range(self.number_of_bits)
        ]

        return new_solution

    @classmethod
    def name(cls) -> str:
        return cls.__name__


# TODO: optimize with numpy, the speed is terrible.
def energy_function(sequence):
    energy = 0
    mapped_seq = list(map(lambda x: -1 if x is True else 1, sequence))
    for distance in range(1, len(sequence)):
        energy += aperiodic_autocorrelation(mapped_seq, distance) ** 2
    return energy


def aperiodic_autocorrelation(sequence, distance):
    autocorr = 0
    for i in range(0, len(sequence) - distance):
        autocorr += sequence[i] * sequence[i + distance]
    return autocorr


def merit_factor(sequence):
    return len(sequence) ** 2 / (2 * energy_function(sequence))
