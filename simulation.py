from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection

from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import SimpleRandomMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Sphere, Rastrigin

from agents import BaseAgent, AgentWithTrust
from runner import Runner
from benchmark_problems import ExpandedSchaffer, Griewank

if __name__ == "__main__":
    # Output file prep
    if not os.path.exists("./output"):
        os.makedirs("./output")
    
    for _ in range(10):  # Number of tests
        
        now = datetime.datetime.now()
        current_date = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        )

        NUM_OF_VARS = 100
        for problem in [
            Sphere(NUM_OF_VARS),
            Rastrigin(NUM_OF_VARS),
            ExpandedSchaffer(NUM_OF_VARS),
            Griewank(NUM_OF_VARS),
        ]:
            for agent_class in [BaseAgent, AgentWithTrust]:
                output_file_name = f"{agent_class.__name__}_{problem.__class__.__name__}_{current_date}.csv"

                with open("output/" + output_file_name, "wt") as f:
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
                        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
                        output_file=f,
                    )
                    runner.run_simulation()

                    print(
                        f"Best result for {agent_class.__name__} for {problem.__class__.__name__} problem:",
                        max(
                            agent.algorithm.get_result().objectives[0]
                            for agent in runner.agents
                        ),
                    )
