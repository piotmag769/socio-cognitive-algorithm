from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection

from jmetal.operator.crossover import SBXCrossover, SPXCrossover
from jmetal.operator.mutation import SimpleRandomMutation, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm import Runner, StrategyAgent
from analysis.constants_and_params import (
    ACCEPT_STRATEGIES_TO_TEST,
    AGENTS_TO_TEST,
    OUTPUT_DIR,
    PROBLEMS_TO_TEST,
    SEND_STRATEGIES_TO_TEST,
)

NUMBER_OF_RUNS = 10
NUM_OF_VARS = 100


def run_simulations_and_save_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for _ in range(NUMBER_OF_RUNS):
        now = datetime.datetime.now()
        current_date = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        )

        for problem in [problem_type(NUM_OF_VARS) for problem_type in PROBLEMS_TO_TEST]:
            for agent_class in AGENTS_TO_TEST:
                if agent_class is StrategyAgent:
                    for accept_strategy in ACCEPT_STRATEGIES_TO_TEST:
                        for send_strategy in SEND_STRATEGIES_TO_TEST:
                            output_file_path = f"{OUTPUT_DIR}/{agent_class.name()}_{accept_strategy}_{send_strategy}_{problem.name()}_{current_date}.csv"
                            run_single_simulation(
                                agent_class,
                                problem,
                                output_file_path,
                                accept_strategy,
                                send_strategy,
                            )
                else:
                    output_file_path = f"{OUTPUT_DIR}/{agent_class.name()}_{problem.name()}_{current_date}.csv"
                    run_single_simulation(
                        agent_class, problem, output_file_path, None, None
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
        # For discrete problems.
        mutation=BitFlipMutation(0.5),
        crossover=SPXCrossover(0.5),
        # For continuous problems.
        # mutation=SimpleRandomMutation(0.5),
        # crossover=SBXCrossover(0.5),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
        send_strategy=send_strategy,
        accept_strategy=accept_strategy,
    )
    runner.run_simulation()

    print(
        f"Best result for {output_file_path}:",
        max(agent.algorithm.result().objectives[0] for agent in runner.agents),
    )


if __name__ == "__main__":
    run_simulations_and_save_results()
