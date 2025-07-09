import os
import re

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from .constants_and_params import (
    OUTPUT_DIR,
    SIGNIFICANCE_LEVEL,
    PROBLEMS_TO_TEST,
)

# def conduct_wilcoxon_tests():
#     for problem in PROBLEMS_TO_TEST:
#         algorithms_best_results = []
#         for algorithm in ["BaseAgent", "CustomMultiClass"]:
#             regex = rf".*{algorithm}_{problem.name()}.*\.csv"
#             find_and_add_best_results(
#                 algorithms_best_results, regex, algorithm, problem.name()
#             )
#         for i in range(len(algorithms_best_results)):
#             for j in range(i + 1, len(algorithms_best_results)):
#                 if (
#                     wilcoxon(
#                         algorithms_best_results[i][1], algorithms_best_results[j][1]
#                     ).pvalue
#                     <= SIGNIFICANCE_LEVEL
#                 ):
#                     print(
#                         f"STATISTICALLY DIFFERENT {algorithms_best_results[i][0]} {algorithms_best_results[j][0]} for: {problem.name()}"
#                     )
#                 # else:
#                 #     print(
#                 #         f"statistically the same {algorithms_best_results[i][0]} {algorithms_best_results[j][0]} for: {problem.name()}"
#                 #     )


# def find_and_add_best_results(algorithms_best_results, regex, algorithm, problem):
#     best_results_for_agent = []
#     dir = f"{OUTPUT_DIR}/base/{problem}"
#     for filename in os.listdir(path=dir):
#         if re.match(regex, filename):
#             current_df = pd.read_csv(f"{dir}/{filename}")

#             best_value = current_df["score"].min()
#             best_results_for_agent.append(best_value)
#     algorithms_best_results.append((algorithm, best_results_for_agent))
#     print(np.mean(best_results_for_agent))


def conduct_wilcoxon_tests():
    # Find base best results for each algorithm.
    base_algorithm_results = {}
    for problem in PROBLEMS_TO_TEST:
        dir = f"output/base/{problem.name()}"
        base_algorithm_results[problem.name()] = find_best_results(dir)

    for dir in os.listdir(OUTPUT_DIR):
        for problem in PROBLEMS_TO_TEST:
            base_results = base_algorithm_results[problem.name()]
            algorithms_best_results = find_best_results(
                f"{OUTPUT_DIR}/{dir}/{problem.name()}"
            )

            if (
                wilcoxon(algorithms_best_results, base_results).pvalue
                <= SIGNIFICANCE_LEVEL
            ) and np.mean(algorithms_best_results) <= np.mean(base_results):
                print(f"STATISTICALLY DIFFERENT: {problem.name()} {dir}")
                print(
                    f"BEST: {np.mean(algorithms_best_results)} BASE: {np.mean(base_results)}"
                )
                print(algorithms_best_results)
                print(base_results)


def find_best_results(dir):
    best_results_for_agent = []
    for filename in os.listdir(path=dir):
        current_df = pd.read_csv(f"{dir}/{filename}")
        best_value = current_df["score"].min()
        best_results_for_agent.append(best_value)

    return best_results_for_agent


if __name__ == "__main__":
    conduct_wilcoxon_tests()
