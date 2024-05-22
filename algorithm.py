from functools import cmp_to_key
from copy import deepcopy

import random
import time
from math import ceil
from typing import Sequence

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.solution import Solution
from jmetal.config import store
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import SimpleRandomMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Sphere


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

    def __eq__(self, other):
        # Two different objects are always unequal.
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))


class AgentWithTrust(BaseAgent):
    # Trust level equal to i means that i-th best solution will be shared - the lower, the better.
    MAX_TRUST_LEVEL = 0

    def __init__(self, algorithm: GeneticAlgorithm):
        self.algorithm = algorithm
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


class ExchangeMarket:
    def __init__(self, agents: Sequence[BaseAgent]):
        self.agents = agents

    def exchange_information(self):
        was_paired_map = {}
        for agent in self.agents:
            was_paired_map[agent] = True
            not_paired_agents = [
                ag for ag in self.agents if not was_paired_map.get(ag, False)
            ]

            if len(not_paired_agents) == 0:
                break

            paired_agent = random.choice(not_paired_agents)
            was_paired_map[paired_agent] = True

            agent.use_shared_solutions(
                paired_agent.get_solutions_to_share(agent), paired_agent
            )
            paired_agent.use_shared_solutions(
                agent.get_solutions_to_share(paired_agent), agent
            )


class BaseRunner:
    GENERATIONS_PER_SWAP = 10

    def __init__(
        self,
        agents_number: int,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        solution_comparator: Comparator = ObjectiveComparator(0),
    ):
        self.agents = [
            BaseAgent(
                GeneticAlgorithm(
                    deepcopy(problem),
                    population_size,
                    offspring_population_size,
                    deepcopy(mutation),
                    deepcopy(crossover),
                    deepcopy(selection),
                    deepcopy(termination_criterion),
                    deepcopy(population_generator),
                    deepcopy(population_evaluator),
                    deepcopy(solution_comparator),
                )
            )
            for _ in range(agents_number)
        ]
        self.exchange_market = ExchangeMarket(self.agents)

    def run_simulation(self):
        start_computing_time = time.time()

        for agent in self.agents:
            agent.algorithm.solutions = agent.algorithm.create_initial_solutions()

        for agent in self.agents:
            agent.algorithm.solutions = agent.algorithm.evaluate(
                agent.algorithm.solutions
            )

        for agent in self.agents:
            agent.algorithm.init_progress()

        # TODO: update this to make sense with more compilcated criteria than number of evaluations
        number_of_generations = 0
        while not agent.algorithm.stopping_condition_is_met():
            number_of_generations += 1
            for agent in self.agents:
                agent.algorithm.step()
                agent.algorithm.update_progress()
            if number_of_generations % BaseRunner.GENERATIONS_PER_SWAP == 0:
                self.exchange_market.exchange_information()

        total_computing_time = time.time() - start_computing_time

        for agent in self.agents:
            agent.algorithm.start_computing_time = start_computing_time
            agent.algorithm.total_computing_time = total_computing_time


class RunnerWithTrust(BaseRunner):
    def __init__(
        self,
        agents_number: int,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        solution_comparator: Comparator = ObjectiveComparator(0),
    ):
        self.agents = [
            AgentWithTrust(
                GeneticAlgorithm(
                    deepcopy(problem),
                    population_size,
                    offspring_population_size,
                    deepcopy(mutation),
                    deepcopy(crossover),
                    deepcopy(selection),
                    deepcopy(termination_criterion),
                    deepcopy(population_generator),
                    deepcopy(population_evaluator),
                    deepcopy(solution_comparator),
                )
            )
            for _ in range(agents_number)
        ]
        self.exchange_market = ExchangeMarket(self.agents)


if __name__ == "__main__":
    NUM_OF_ITEMS = 10
    problem = Sphere(
        number_of_variables=NUM_OF_ITEMS,
    )
    runner = BaseRunner(
        agents_number=10,
        problem=deepcopy(problem),
        population_size=50,
        offspring_population_size=30,
        mutation=SimpleRandomMutation(0.5),
        crossover=SBXCrossover(0.9),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
    )
    runner.run_simulation()

    for agent in runner.agents:
        print(agent.algorithm.get_result().objectives)

    runner = RunnerWithTrust(
        agents_number=10,
        problem=problem,
        population_size=50,
        offspring_population_size=30,
        mutation=SimpleRandomMutation(0.5),
        crossover=SBXCrossover(0.9),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
    )
    runner.run_simulation()

    for agent in runner.agents:
        print(agent.algorithm.get_result().objectives)
