# https://en.wikipedia.org/wiki/Test_functions_for_optimization
import numpy as np
import random

from jmetal.core.problem import BinaryProblem, FloatProblem
from jmetal.core.solution import BinarySolution, FloatSolution


class ExpandedSchaffer(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(ExpandedSchaffer, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(X)"]

        self.lower_bound = [-100] * number_of_variables
        self.upper_bound = [100] * number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        x_next = np.roll(x, -1)
        tmp = x**2 + x_next**2
        val = 0.5 + (np.sin(np.sqrt(tmp)) ** 2 - 0.5) / (1 + 0.001 * tmp) ** 2
        solution.objectives[0] = np.sum(val)

        return solution

    def name(self) -> str:
        return "Expanded Schaffer"


class Griewank(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Griewank, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(X)"]

        self.lower_bound = [-600] * number_of_variables
        self.upper_bound = [600] * number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        sigma = np.sum(x**2 / 4000)
        pi_denominators = np.sqrt(np.arange(1, x.shape[0] + 1))
        pi = np.prod(np.cos(np.divide(x, pi_denominators)))
        solution.objectives[0] = sigma - pi + 1

        return solution

    def name(self) -> str:
        return "Griewank"


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

    def name(self):
        return "LABS"


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
