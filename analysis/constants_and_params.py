from algorithm.agents import StrategyAgent, AgentWithTrust
from algorithm.agents.base import BaseAgent
from algorithm.agents.strategy_based import AcceptStrategy, SendStrategy, TrustMechanism
from algorithm.follow_best import FollowBestGA
from algorithm.follow_best_distinct import FollowBestDistinctGA
from algorithm.repel_worst_gravity import RepelWorstGravity
from algorithm.repel_worst_gravity_multistep import RepelWorstGravityMultistep

from problems import LABS

from jmetal.problem.singleobjective.knapsack import Knapsack
from jmetal.problem.singleobjective.tsp import TSP
from jmetal.algorithm.singleobjective import GeneticAlgorithm

OUTPUT_DIR = "./OFFICIAL_OUTPUT"

PLOTS_DIR = "./graphs"
BOX_AND_WHISKERS_PLOTS_DIR = f"{PLOTS_DIR}/box_and_whiskers"
MEAN_PLOTS_DIR = f"{PLOTS_DIR}/mean"
MULTI_CLASS_PLOTS_DIR = f"{PLOTS_DIR}/PERF_200vars_migration_local_trust"

SIGNIFICANCE_LEVEL = 0.05
NUMBER_OF_ITERATIONS = 998  # ~10000 evaluations
ITERATION_INTERVAL = 50
NUMBER_OF_RUNS = 10
POPULATION_SIZE = 20
OFFSPRING_POPULATION_SIZE = 10
GENERATIONS_PER_SWAP = 10
MAX_EVALUATIONS = 10000
MIGRATION = True
AGENTS_NUMBER = 12
POPULATION_PART_TO_SWAP = 0.1
NO_SEND_PENALTY = int(POPULATION_SIZE * POPULATION_PART_TO_SWAP)
ALGORITHM_TYPES = [
    GeneticAlgorithm,
    FollowBestGA,
    FollowBestDistinctGA,
    RepelWorstGravity,
    RepelWorstGravityMultistep,
]

PROBLEMS_TO_TEST = [LABS, Knapsack, TSP("tsp.txt")]

agents = []
send_strategies = []
accept_strategies = []

for send_strategy in [
    SendStrategy.Best,
    SendStrategy.Average,
    SendStrategy.Random,
    SendStrategy.Outlying,
]:
    for accept_strategy in [
        AcceptStrategy.Always,
        AcceptStrategy.Better,
        AcceptStrategy.Different,
    ]:
        agents.append(StrategyAgent)
        send_strategies.append(send_strategy)
        accept_strategies.append(accept_strategy)

MULTI_CLASS_SETUP = [agents, send_strategies, accept_strategies]
