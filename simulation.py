from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection

from jmetal.operator.crossover import SBXCrossover, SPXCrossover
from jmetal.operator.mutation import SimpleRandomMutation, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm import (
    AcceptStrategy,
    Runner,
    SendStrategy,
)

from analysis.constants_and_params import AGENTS_TO_TEST, OUTPUT_DIR, PROBLEMS_TO_TEST

NUMBER_OF_RUNS = 1
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
                output_file_name = (
                    f"{agent_class.name()}_{problem.name()}_{current_date}.csv"
                )

                with open(f"{OUTPUT_DIR}/{output_file_name}", "w") as f:
                    runner = Runner(
                        agent_class=agent_class,
                        agents_number=10,
                        generations_per_swap=10,
                        problem=deepcopy(problem),
                        population_size=20,
                        offspring_population_size=10,
                        # For discrete problems.
                        mutation=BitFlipMutation(0.5),
                        crossover=SPXCrossover(0.5),
                        # For continuous problem.
                        # mutation=SimpleRandomMutation(0.5),
                        # crossover=SBXCrossover(0.5),
                        selection=BinaryTournamentSelection(),
                        termination_criterion=StoppingByEvaluations(
                            max_evaluations=10000
                        ),
                        send_strategy=SendStrategy.Random,
                        accept_strategy=AcceptStrategy.Better,
                        output_file=f,
                    )
                    runner.run_simulation()

                    print(
                        f"Best result for {agent_class.name()} for {problem.name()} problem:",
                        max(
                            agent.algorithm.result().objectives[0]
                            for agent in runner.agents
                        ),
                    )


if __name__ == "__main__":
    run_simulations_and_save_results()
