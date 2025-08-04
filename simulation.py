from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection

from jmetal.operator.crossover import SPXCrossover, PMXCrossover
from jmetal.operator.mutation import BitFlipMutation, PermutationSwapMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.problem.singleobjective.tsp import TSP
from jmetal.problem.singleobjective.knapsack import Knapsack
from algorithm import Runner
from algorithm.agents.base import BaseAgent
from algorithm.agents.strategy_based import TrustMechanism
from analysis.constants_and_params import (
    OUTPUT_DIR,
    MULTI_CLASS_SETUP,
    NUMBER_OF_RUNS,
    MIGRATION,
    POPULATION_SIZE,
    OFFSPRING_POPULATION_SIZE,
    GENERATIONS_PER_SWAP,
    MAX_EVALUATIONS,
    AGENTS_NUMBER,
    NO_SEND_PENALTY,
    POPULATION_PART_TO_SWAP,
    ALGORITHM_TYPES,
)
from knapsack import PROFITS, WEIGHTS
from problems import LABS
from jmetal.algorithm.singleobjective import GeneticAlgorithm

# Multi class setup parsing
agents = MULTI_CLASS_SETUP[0]
send_strategies = MULTI_CLASS_SETUP[1]
accept_strategies = MULTI_CLASS_SETUP[2]


def run_simulations_and_save_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.datetime.now()

    for starting_trust in [0, 10, 20]:
        for trust_mechanism in [TrustMechanism.Local, TrustMechanism.Global]:
            for _ in range(NUMBER_OF_RUNS):
                now = datetime.datetime.now()
                current_date = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

                for algorithm_type in ALGORITHM_TYPES:
                    for problem in [
                        TSP("tsp.txt"),
                        LABS(100),
                        Knapsack(
                            number_of_items=100,
                            capacity=1000,
                            weights=deepcopy(WEIGHTS),
                            profits=deepcopy(PROFITS),
                        ),
                    ]:
                        # """BASE AGENT CLASS SIMULATION"""
                        # dir = f"{OUTPUT_DIR}/{problem.name()}/{BaseAgent.name()}/{algorithm_type.__name__}"
                        # os.makedirs(dir, exist_ok=True)
                        # output_file_path = f"{dir}/{BaseAgent.name()}_{problem.name()}_migration_{algorithm_type.__name__}_{current_date}_.csv"
                        # run_single_simulation(
                        #     BaseAgent,
                        #     problem,
                        #     output_file_path,
                        #     None,
                        #     None,
                        #     algorithm_type,
                        #     MIGRATION,
                        #     None,
                        #     None,
                        # )

                        """MULTI AGENT CLASS SIMULATION"""
                        dir = f"{OUTPUT_DIR}/{problem.name()}/AllMultiClass/{algorithm_type.__name__}/{trust_mechanism}/starting_trust={starting_trust}"
                        os.makedirs(dir, exist_ok=True)
                        output_file_path = f"{dir}/AllMultiClass_{problem.name()}_migration_{trust_mechanism}_{algorithm_type.__name__}_starting_trust={starting_trust}_{current_date}.csv"
                        run_single_simulation(
                            agents,
                            problem,
                            output_file_path,
                            accept_strategies,
                            send_strategies,
                            algorithm_type,
                            MIGRATION,
                            starting_trust,
                            trust_mechanism,
                        )


from jmetal.core.problem import BinaryProblem


def run_single_simulation(
    agent_class,
    problem,
    output_file_path,
    accept_strategy,
    send_strategy,
    algorithm_type,
    migration,
    starting_trust,
    trust_mechanism,
):
    mutation = (
        BitFlipMutation(0.5)
        if isinstance(problem, BinaryProblem)
        else PermutationSwapMutation(0.5)
    )
    crossover = (
        SPXCrossover(0.5) if isinstance(problem, BinaryProblem) else PMXCrossover(0.5)
    )

    print(output_file_path)
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
        migration=migration,
        trust_mechanism=trust_mechanism,
        starting_trust=starting_trust,
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
