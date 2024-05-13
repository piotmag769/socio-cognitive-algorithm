from functools import cmp_to_key
from copy import deepcopy

import random
import time

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
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.knapsack import Knapsack


class Agent:
    # Trust level equal to i means that i-th best solution will be shared - the lower, the better.
    MAX_TRUST_LEVEL = 0

    def __init__(self, algorithm: GeneticAlgorithm):
        self.algorithm = algorithm
        self.trust_level_map = {}

    def get_solution_to_share(self, agent_to_share_with) -> Solution:
        if agent_to_share_with not in self.trust_level_map:
            self.trust_level_map[agent_to_share_with] = Agent.MAX_TRUST_LEVEL

        trust_lvl = self.trust_level_map[agent_to_share_with]
        max_solution_index = len(self.algorithm.solutions) - 1
        index_of_solution_to_share = (
            trust_lvl if trust_lvl <= max_solution_index else max_solution_index
        )

        return self.algorithm.solutions[index_of_solution_to_share]

    def use_shared_solution(self, shared_solution, agent_sharing_the_solution):
        self.algorithm.solutions.append(shared_solution)
        # Compare solutions (we assume they were already evaluated).
        self.algorithm.solutions.sort(
            key=cmp_to_key(self.algorithm.solution_comparator.compare)
        )
        # Get best solutions.
        self.algorithm.solutions = self.algorithm.solutions[
            : self.algorithm.population_size
        ]

        if agent_sharing_the_solution not in self.trust_level_map:
            self.trust_level_map[agent_sharing_the_solution] = Agent.MAX_TRUST_LEVEL

        if shared_solution in self.algorithm.solutions:
            # A useful solution was shared.
            self.trust_level_map[agent_sharing_the_solution] = max(
                Agent.MAX_TRUST_LEVEL,
                self.trust_level_map[agent_sharing_the_solution] - 1,
            )
        else:
            # A useless solution was shared.
            self.trust_level_map[agent_sharing_the_solution] += 1

    def __eq__(self, other):
        # Two different objects are always unequal.
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))


class ExchangeMarket:
    def __init__(self, agents: list[Agent]):
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

            agent.use_shared_solution(
                paired_agent.get_solution_to_share(agent), paired_agent
            )
            paired_agent.use_shared_solution(
                agent.get_solution_to_share(paired_agent), agent
            )


class Runner:
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
            Agent(
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
        while not agent.algorithm.stopping_condition_is_met():
            for agent in self.agents:
                agent.algorithm.step()
                agent.algorithm.update_progress()
            self.exchange_market.exchange_information()

        total_computing_time = time.time() - start_computing_time

        for agent in self.agents:
            agent.algorithm.start_computing_time = start_computing_time
            agent.algorithm.total_computing_time = total_computing_time


if __name__ == "__main__":
    NUM_OF_ITEMS = 20
    problem = Knapsack(
        number_of_items=NUM_OF_ITEMS,
        capacity=10,
        weights=[random.randint(1, 10) for _ in range(NUM_OF_ITEMS)],
        profits=[random.randint(1, 10) for _ in range(NUM_OF_ITEMS)],
    )
    print("WEIGHTS:", problem.weights)
    print("PROFITS:", problem.profits)
    runner = Runner(
        agents_number=10,
        problem=problem,
        population_size=2,
        offspring_population_size=100,
        mutation=BitFlipMutation(0.5),
        crossover=SPXCrossover(0.9),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
    )
    runner.run_simulation()

    for agent in runner.agents:
        print(agent.algorithm.get_result())
