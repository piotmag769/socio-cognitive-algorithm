import os
import re

import pandas as pd

from scipy.stats import wilcoxon

from algorithm.agents.strategy_based import StrategyAgent

from .constants_and_params import (
    ACCEPT_STRATEGIES_TO_TEST,
    OUTPUT_DIR,
    SEND_STRATEGIES_TO_TEST,
    SIGNIFICANCE_LEVEL,
    PROBLEMS_TO_TEST,
    AGENTS_TO_TEST,
)


def conduct_wilcoxon_tests():
    for problem in PROBLEMS_TO_TEST:
        algorithms_best_results = []
        for agent in AGENTS_TO_TEST:
            if agent is StrategyAgent:
                for accept_strategy in ACCEPT_STRATEGIES_TO_TEST:
                    for send_strategy in SEND_STRATEGIES_TO_TEST:
                        regex = rf".*{agent.name()}_{accept_strategy}_{send_strategy}_{problem.name()}.*\.csv"
                        algorithm_name = (
                            f"{agent.name()}_{accept_strategy}_{send_strategy}"
                        )
                        find_and_add_best_results(
                            algorithms_best_results, regex, algorithm_name
                        )
            else:
                regex = rf".*{agent.name()}_{problem.name()}.*\.csv"
                algorithm_name = agent.name()
                find_and_add_best_results(
                    algorithms_best_results, regex, algorithm_name
                )

        for i in range(len(algorithms_best_results)):
            for j in range(i + 1, len(algorithms_best_results)):
                if (
                    wilcoxon(
                        algorithms_best_results[i][1], algorithms_best_results[j][1]
                    ).pvalue
                    <= SIGNIFICANCE_LEVEL
                ):
                    print(
                        f"STATISTICALLY DIFFERENT {algorithms_best_results[i][0]} {algorithms_best_results[j][0]} for: {problem.name()}"
                    )
                # else:
                #     print(
                #         f"statistically the same {algorithms_best_results[i][0]} {algorithms_best_results[j][0]} for: {problem.name()}"
                #     )


def find_and_add_best_results(algorithms_best_results, regex, algorithm_name):
    best_results_for_agent = []
    for filename in os.listdir(path=OUTPUT_DIR):
        if re.match(regex, filename):
            current_df = pd.read_csv(f"{OUTPUT_DIR}/{filename}")

            best_value = current_df["score"].min()
            best_results_for_agent.append(best_value)
    algorithms_best_results.append((algorithm_name, best_results_for_agent))


if __name__ == "__main__":
    conduct_wilcoxon_tests()
