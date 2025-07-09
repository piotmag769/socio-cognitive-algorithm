from copy import deepcopy
import datetime, os
import random

from jmetal.operator import BinaryTournamentSelection
from jmetal.core.problem import BinaryProblem

from jmetal.operator.crossover import SBXCrossover, SPXCrossover
from jmetal.operator.mutation import SimpleRandomMutation, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.singleobjective import GeneticAlgorithm


from jmetal.problem.singleobjective.knapsack import Knapsack
from algorithm import Runner
from algorithm.agents.base import BaseAgent
from analysis.constants_and_params import (
    OUTPUT_DIR,
    MULTI_CLASS_SETUP,
    TRUST_MECHANISM,
    NUMBER_OF_RUNS,
    NUM_OF_VARS,
    MIGRATION,
    POPULATION_SIZE,
    OFFSPRING_POPULATION_SIZE,
    GENERATIONS_PER_SWAP,
    MAX_EVALUATIONS,
    AGENTS_NUMBER,
    STARTING_TRUST,
    NO_SEND_PENALTY,
    POPULATION_PART_TO_SWAP,
    ALGORITHM_TYPES,
)
from knapsack import PROFITS, WEIGHTS
from problems import LABS

# Multi class setup parsing
agents = MULTI_CLASS_SETUP[0]
send_strategies = MULTI_CLASS_SETUP[1]
accept_strategies = MULTI_CLASS_SETUP[2]


def run_simulations_and_save_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.datetime.now()
    start_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    migration_mechanism = "migration" if MIGRATION else "cloning"

    for _ in range(NUMBER_OF_RUNS):
        now = datetime.datetime.now()
        current_date = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        )

        for algorithm_type in [GeneticAlgorithm]:
            for problem in [
                # LABS(NUM_OF_VARS),
                Knapsack(
                    number_of_items=100,
                    capacity=1000,
                    weights=deepcopy(WEIGHTS),
                    profits=deepcopy(PROFITS),
                ),
            ]:
                custom_output = f"{OUTPUT_DIR}/{migration_mechanism}_{TRUST_MECHANISM}_{algorithm_type.__name__}_starting_trust={STARTING_TRUST}_{start_date}"

                dir = f"{OUTPUT_DIR}/Base_{algorithm_type.__name__}"
                os.makedirs(dir, exist_ok=True)
                """BASE AGENT CLASS SIMULATION"""
                output_file_path = f"{dir}/{BaseAgent.name()}_{problem.name()}_{current_date}_{algorithm_type.__name__}.csv"
                run_single_simulation(
                    BaseAgent, problem, output_file_path, None, None, algorithm_type
                )

                """SINGLE AGENT CLASS SIMULATION"""
                # for agent_class in AGENTS_TO_TEST:
                #     if agent_class is StrategyAgent:
                #         for accept_strategy in ACCEPT_STRATEGIES_TO_TEST:
                #             for send_strategy in SEND_STRATEGIES_TO_TEST:
                #                 output_file_path = f"{OUTPUT_DIR}/{agent_class.name()}_{accept_strategy}_{send_strategy}_{problem.name()}_{current_date}.csv"
                #                 run_single_simulation(
                #                     agent_class,
                #                     problem,
                #                     output_file_path,
                #                     accept_strategy,
                #                     send_strategy,
                #                 )
                #     else:
                #         output_file_path = f"{OUTPUT_DIR}/{agent_class.name()}_{problem.name()}_{current_date}.csv"
                #         run_single_simulation(
                #             agent_class, problem, output_file_path, None, None
                #         )

                # """ MULTI AGENT CLASS SIMULATION """
                # dir = f"{custom_output}/{problem.name()}"
                # os.makedirs(dir, exist_ok=True)
                # output_file_path = (
                #     f"{dir}/CustomMultiClass_{problem.name()}_{current_date}.csv"
                # )
                # run_single_simulation(
                #     agents,
                #     problem,
                #     output_file_path,
                #     accept_strategies,
                #     send_strategies,
                #     algorithm_type,
                # )


def run_single_simulation(
    agent_class,
    problem,
    output_file_path,
    accept_strategy,
    send_strategy,
    algorithm_type,
):
    print(output_file_path)
    mutation = (
        BitFlipMutation(0.5)
        if isinstance(problem, BinaryProblem)
        else SimpleRandomMutation(0.5)
    )
    crossover = (
        SPXCrossover(0.5) if isinstance(problem, BinaryProblem) else SBXCrossover(0.5)
    )

    runner = Runner(
        output_file_path=output_file_path,
        agent_class=agent_class,
        agents_number=AGENTS_NUMBER,  # Needed only for single class, non strategy based agents
        generations_per_swap=GENERATIONS_PER_SWAP,
        problem=deepcopy(problem),
        population_size=POPULATION_SIZE,
        offspring_population_size=OFFSPRING_POPULATION_SIZE,
        mutation=mutation,
        crossover=crossover,
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=MAX_EVALUATIONS),
        send_strategy=send_strategy,
        accept_strategy=accept_strategy,
        migration=MIGRATION,
        trust_mechanism=TRUST_MECHANISM,
        starting_trust=STARTING_TRUST,
        no_send_penalty=NO_SEND_PENALTY,
        part_to_swap=POPULATION_PART_TO_SWAP,
        algorithm_type=algorithm_type,
    )
    runner.run_simulation()

    print(
        f"Best result for {output_file_path}:",
        min(agent.algorithm.result().objectives[0] for agent in runner.agents),
    )


if __name__ == "__main__":
    run_simulations_and_save_results()
