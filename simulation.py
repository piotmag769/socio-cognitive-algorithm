from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection

from jmetal.operator.crossover import SBXCrossover, SPXCrossover
from jmetal.operator.mutation import SimpleRandomMutation, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.singleobjective import GeneticAlgorithm

from algorithm import (
    Runner,
    StrategyAgent,
    AgentWithTrust,
    AcceptStrategy,
    SendStrategy,
    BaseAgent,
)
from analysis.constants_and_params import (
    ACCEPT_STRATEGIES_TO_TEST,
    AGENTS_TO_TEST,
    OUTPUT_DIR,
    PROBLEMS_TO_TEST,
    SEND_STRATEGIES_TO_TEST,
    MULTI_CLASS_SETUP,
)

# Multi class setup parsing
agents = MULTI_CLASS_SETUP[0]
send_strategies = MULTI_CLASS_SETUP[1]
accept_strategies = MULTI_CLASS_SETUP[2]

NUMBER_OF_RUNS = 5
NUM_OF_VARS = 100
MIGRATION = True


def run_simulations_and_save_results():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.datetime.now()
    start_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    custom_output = OUTPUT_DIR + "/" + start_date

    for _ in range(NUMBER_OF_RUNS):
        now = datetime.datetime.now()
        current_date = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        )

        for problem in [problem_type(NUM_OF_VARS) for problem_type in PROBLEMS_TO_TEST]:

            ### SINGLE AGENT CLASS SIMULATION
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

            ### MULTI AGENT CLASS SIMULATION
            os.makedirs(custom_output, exist_ok=True)
            output_file_path = (
                f"{custom_output}/CustomMultiClass_{problem.name()}_{current_date}.csv"
            )
            run_single_simulation(
                agents, problem, output_file_path, accept_strategies, send_strategies
            )


def run_single_simulation(
    agent_class, problem, output_file_path, accept_strategy, send_strategy
):
    print(output_file_path)
    runner = Runner(
        output_file_path=output_file_path,
        agent_class=agent_class,
        agents_number=10,
        generations_per_swap=10,
        problem=deepcopy(problem),
        population_size=20,
        offspring_population_size=10,
        ### For discrete problems.
        mutation=BitFlipMutation(0.5),
        crossover=SPXCrossover(0.5),
        ### For continuous problems.
        # mutation=SimpleRandomMutation(0.5),
        # crossover=SBXCrossover(0.5),
        ###
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=100000),
        send_strategy=send_strategy,
        accept_strategy=accept_strategy,
        migration=MIGRATION,
    )
    runner.run_simulation()

    print(
        f"Best result for {output_file_path}:",
        max(agent.algorithm.result().objectives[0] for agent in runner.agents),
    )


if __name__ == "__main__":
    run_simulations_and_save_results()
