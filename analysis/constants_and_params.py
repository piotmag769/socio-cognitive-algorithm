from algorithm.agents import StrategyAgent, AgentWithTrust
from algorithm.agents.base import BaseAgent
from algorithm.agents.strategy_based import AcceptStrategy, SendStrategy, TrustMechanism
from problems import LABS, ExpandedSchaffer, Griewank

OUTPUT_DIR = "./new_output"

PLOTS_DIR = "./graphs"
BOX_AND_WHISKERS_PLOTS_DIR = f"{PLOTS_DIR}/box_and_whiskers"
MEAN_PLOTS_DIR = f"{PLOTS_DIR}/mean"
MULTI_CLASS_PLOTS_DIR = f"{PLOTS_DIR}/PERF_200vars_migration_local_trust"

SIGNIFICANCE_LEVEL = 0.05
NUMBER_OF_ITERATIONS = 9998  # 9998 for 100000 evaluations, 998 for 10000 evaluations
ITERATION_INTERVAL = 50
TRUST_MECHANISM = TrustMechanism.Local
NUMBER_OF_RUNS = 10
NUM_OF_VARS = 200
POPULATION_SIZE = 20
OFFSPRING_POPULATION_SIZE = 10
STARTING_TRUST = 10
GENERATIONS_PER_SWAP = 10
MAX_EVALUATIONS = 100000
MIGRATION = True
AGENTS_NUMBER = 24
POPULATION_PART_TO_SWAP = 0.1
NO_SEND_PENALTY = int(POPULATION_SIZE * POPULATION_PART_TO_SWAP)

AGENTS_TO_TEST = [BaseAgent, StrategyAgent]
# TODO: add Ackley and other binary problem
PROBLEMS_TO_TEST = [
    LABS,
    ExpandedSchaffer,
    Griewank,
]
ACCEPT_STRATEGIES_TO_TEST = [
    strategy
    for strategy in AcceptStrategy  # if strategy is not AcceptStrategy.Different
]
SEND_STRATEGIES_TO_TEST = [
    strategy for strategy in SendStrategy  # if strategy is not SendStrategy.Outlying
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

# CUSTOM MULTI CLASS CONFIG (leave one config uncommented if you want to run it)
agents = []
send_strategies = []
accept_strategies = []

""" 5Creative_3Trust_1Perf_1Solo """
# for _ in range(3):  # Trust Agents
#     agents.append(AgentWithTrust)
#     send_strategies.append(None)
#     accept_strategies.append(None)
# for _ in range(1):  # Solo Agent
#     agents.append(StrategyAgent)
#     send_strategies.append(SendStrategy.Dont)
#     accept_strategies.append(AcceptStrategy.Reject)
# for _ in range(5):  # Creative Agents
#     agents.append(StrategyAgent)
#     send_strategies.append(SendStrategy.Outlying)
#     accept_strategies.append(AcceptStrategy.Different)
# for _ in range(1):  # Perfectionist Agents
#     agents.append(StrategyAgent)
#     send_strategies.append(SendStrategy.Best)
#     accept_strategies.append(AcceptStrategy.Better)

""" 1Extractor_2Tryhard_2Filter_6Creative """
# for _ in range(1):
#     agents.append(StrategyAgent)  # Extractor
#     send_strategies.append(SendStrategy.Dont)
#     accept_strategies.append(AcceptStrategy.Better)
# for _ in range(2):
#     agents.append(StrategyAgent)  # Tryhard
#     send_strategies.append(SendStrategy.Best)
#     accept_strategies.append(AcceptStrategy.Always)
# for _ in range(2):
#     agents.append(StrategyAgent)  # Filter
#     send_strategies.append(SendStrategy.Best)
#     accept_strategies.append(AcceptStrategy.Different)
# for _ in range(6):
#     agents.append(StrategyAgent)  # Creative
#     send_strategies.append(SendStrategy.Outlying)
#     accept_strategies.append(AcceptStrategy.Different)

"""All different mixes."""
for send_strategy in SendStrategy:
    for accept_strategy in AcceptStrategy:
        agents.append(StrategyAgent)
        send_strategies.append(send_strategy)
        accept_strategies.append(accept_strategy)

MULTI_CLASS_SETUP = [agents, send_strategies, accept_strategies]
