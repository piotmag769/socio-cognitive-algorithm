# https://en.wikipedia.org/wiki/Test_functions_for_optimization
import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


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
