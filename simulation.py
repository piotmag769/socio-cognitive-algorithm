from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection

from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import SimpleRandomMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Sphere, Rastrigin

from algorithm import (
    AcceptStrategy,
    AgentWithTrust,
    BaseAgent,
    Runner,
    SendStrategy,
    StrategyAgent,
)

from analysis.constants import OUTPUT_DIR
from problems import ExpandedSchaffer, Griewank

NUMBER_OF_RUNS = 1
NUM_OF_VARS = 100


def run_simulations_and_save_results():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for _ in range(NUMBER_OF_RUNS):
        now = datetime.datetime.now()
        current_date = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        )

        for problem in [
            Sphere(NUM_OF_VARS),
            Rastrigin(NUM_OF_VARS),
            ExpandedSchaffer(NUM_OF_VARS),
            Griewank(NUM_OF_VARS),
        ]:
            for agent_class in [BaseAgent, AgentWithTrust, StrategyAgent]:
                output_file_name = f"{agent_class.__name__}_{problem.__class__.__name__}_{current_date}.csv"

                with open(f"{OUTPUT_DIR}/{output_file_name}", "w") as f:
                    runner = Runner(
                        agent_class=agent_class,
                        agents_number=10,
                        generations_per_swap=10,
                        problem=deepcopy(problem),
                        population_size=20,
                        offspring_population_size=10,
                        mutation=SimpleRandomMutation(0.5),
                        crossover=SBXCrossover(0.5),
                        selection=BinaryTournamentSelection(),
                        termination_criterion=StoppingByEvaluations(
                            max_evaluations=10000
                        ),
                        send_strategy=SendStrategy.Outlying,
                        accept_strategy=AcceptStrategy.Different,
                        output_file=f,
                    )
                    runner.run_simulation()

                    print(
                        f"Best result for {agent_class.__name__} for {problem.__class__.__name__} problem:",
                        max(
                            agent.algorithm.result().objectives[0]
                            for agent in runner.agents
                        ),
                    )


if __name__ == "__main__":
    run_simulations_and_save_results()
