from algorithm.agents import BaseAgent, StrategyAgent
from algorithm.agents.strategy_based import AcceptStrategy, SendStrategy
from problems import LABS

OUTPUT_DIR = "./output"

PLOTS_DIR = "./graphs"
BOX_AND_WHISKERS_PLOTS_DIR = f"{PLOTS_DIR}/box_and_whiskers"
MEAN_PLOTS_DIR = f"{PLOTS_DIR}/mean"

SIGNIFICANCE_LEVEL = 0.05
NUMBER_OF_ITERATIONS = 998

AGENTS_TO_TEST = [StrategyAgent]
# Change this to test for continuous problems.
PROBLEMS_TO_TEST = [LABS]
ACCEPT_STRATEGIES_TO_TEST = [
    strategy for strategy in AcceptStrategy if strategy is not AcceptStrategy.Different
]
SEND_STRATEGIES_TO_TEST = [
    strategy for strategy in SendStrategy if strategy is not SendStrategy.Outlying
]

# TODO: get rid of this and use `AGENTS_TO_TEST` and `PROBLEMS_TO_TEST` directly.
# Experiment names order matters!!!
# It's used later for plotting order.
# Group the names by problem type and have
# the order of agents consistent between
# the problem types.
EXPERIMENTS = []
for problem in PROBLEMS_TO_TEST:
    for agent in AGENTS_TO_TEST:
        if agent is StrategyAgent:
            for accept_strategy in ACCEPT_STRATEGIES_TO_TEST:
                for send_strategy in SEND_STRATEGIES_TO_TEST:
                    EXPERIMENTS.append(
                        f"{agent.name()}_{accept_strategy}_{send_strategy}_{problem.name()}"
                    )
        else:
            EXPERIMENTS.append(f"{agent.name()}_{problem.name()}")
