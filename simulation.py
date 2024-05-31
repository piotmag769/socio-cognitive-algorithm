from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection

from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import SimpleRandomMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Sphere

from agents import BaseAgent, AgentWithTrust
from runner import Runner

if __name__ == "__main__":
    # Output file prep
    if not os.path.exists("./output"):
        os.makedirs("./output")
    now = datetime.datetime.now()
    current_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )

    for agent_class in [BaseAgent, AgentWithTrust]:
        output_file_name = f"{agent_class.__name__}_" + current_date + ".csv"

        with open("output/" + output_file_name, "wt") as f:
            problem = Sphere(
                number_of_variables=50,
            )
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
                f"Best result for {agent_class.__name__}:",
                max(
                    agent.algorithm.get_result().objectives[0]
                    for agent in runner.agents
                ),
            )
