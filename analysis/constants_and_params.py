from algorithm.agents import BaseAgent, StrategyAgent, AgentWithTrust
from algorithm.agents.strategy_based import AcceptStrategy, SendStrategy
from problems import LABS, Griewank

OUTPUT_DIR = "./output"

PLOTS_DIR = "./graphs"
BOX_AND_WHISKERS_PLOTS_DIR = f"{PLOTS_DIR}/box_and_whiskers"
MEAN_PLOTS_DIR = f"{PLOTS_DIR}/mean"

SIGNIFICANCE_LEVEL = 0.05
NUMBER_OF_ITERATIONS = 998

AGENTS_TO_TEST = [AgentWithTrust, StrategyAgent]
# Change this to test for continuous problems.
PROBLEMS_TO_TEST = [Griewank]
ACCEPT_STRATEGIES_TO_TEST = [
    strategy for strategy in AcceptStrategy # if strategy is not AcceptStrategy.Different
]
SEND_STRATEGIES_TO_TEST = [
    strategy for strategy in SendStrategy # if strategy is not SendStrategy.Outlying
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

# CUSTOM MULTI CLASS CONFIG
agents = []
send_strategies = []
accept_strategies = []
for _ in range(4):  # Trust Agents
    agents.append(AgentWithTrust)
    send_strategies.append(None)
    accept_strategies.append(None)
for _ in range(1):  # Solo Agent
    agents.append(StrategyAgent)
    send_strategies.append(SendStrategy.Dont)
    accept_strategies.append(AcceptStrategy.Reject)
for _ in range(3):  # Creative Agents
    agents.append(StrategyAgent)
    send_strategies.append(SendStrategy.Outlying)
    accept_strategies.append(AcceptStrategy.Different)
for _ in range(2):  # Perfectionist Agents
    agents.append(StrategyAgent)
    send_strategies.append(SendStrategy.Best)
    accept_strategies.append(AcceptStrategy.Better)
MULTI_CLASS_SETUP = [agents, send_strategies, accept_strategies]
