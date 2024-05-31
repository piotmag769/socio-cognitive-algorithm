import time
from typing import Callable, Type, Optional
from io import TextIOWrapper

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

from agents import BaseAgent
from exchange_logic import ExchangeMarket


class Runner:
    def __init__(
        self,
        agent_class: Callable[[GeneticAlgorithm], Type[BaseAgent]],
        agents_number: int,
        generations_per_swap: int,
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
        output_file: Optional[TextIOWrapper] = None,
    ):
        self.agents = [
            agent_class(
                GeneticAlgorithm(
                    problem,
                    population_size,
                    offspring_population_size,
                    mutation,
                    crossover,
                    selection,
                    termination_criterion,
                    population_generator,
                    population_evaluator,
                    solution_comparator,
                )
            )
            for _ in range(agents_number)
        ]
        self.exchange_market = ExchangeMarket(self.agents)
        self.generations_per_swap = generations_per_swap
        self.output_file = output_file

    def note_progress(self, agent, agent_id, gen_nr):
        # If it's the first record, write column descriptors
        if gen_nr + agent_id == 1:
            self.output_file.write(f"generation,agent_id,score\n")
        # Write down important statistics
        score = agent.algorithm.get_result().objectives[0]
        self.output_file.write(f"{gen_nr}, {agent_id}, {score}\n")

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
            for agent_id, agent in enumerate(self.agents):
                agent.algorithm.step()
                agent.algorithm.update_progress()
                if self.output_file is not None:
                    self.note_progress(agent, agent_id, number_of_generations)
            if number_of_generations % self.generations_per_swap == 0:
                self.exchange_market.exchange_information()

        total_computing_time = time.time() - start_computing_time

        for agent in self.agents:
            agent.algorithm.start_computing_time = start_computing_time
            agent.algorithm.total_computing_time = total_computing_time
