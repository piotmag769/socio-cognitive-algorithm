import os

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from .constants_and_params import (
    OUTPUT_DIR,
    SIGNIFICANCE_LEVEL,
    PROBLEMS_TO_TEST,
)


def conduct_wilcoxon_tests():
    for problem in PROBLEMS_TO_TEST:
        problem_dir = f"./{OUTPUT_DIR}/{problem.name()}"
        base_dir = f"{problem_dir}/BaseAgent/GeneticAlgorithm"
        base_results = find_best_results(base_dir)

        for root, dirs, _ in os.walk(problem_dir):
            if len(dirs) == 0 and base_dir != root:
                algorithms_best_results = find_best_results(root)

                if (
                    wilcoxon(algorithms_best_results, base_results).pvalue
                    <= SIGNIFICANCE_LEVEL
                ) and np.mean(algorithms_best_results) <= np.mean(base_results):
                    print(f"STATISTICALLY DIFFERENT: {root}")
                    print(
                        f"BEST: {np.mean(algorithms_best_results)} BASE: {np.mean(base_results)}"
                    )


def find_best_results(dir):
    best_results_for_agent = []
    for filename in os.listdir(path=dir):
        current_df = pd.read_csv(f"{dir}/{filename}")
        best_value = current_df["score"].min()
        best_results_for_agent.append(best_value)

    return best_results_for_agent


if __name__ == "__main__":
    conduct_wilcoxon_tests()
