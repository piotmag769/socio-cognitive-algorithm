import os
import re

import pandas as pd

from scipy.stats import wilcoxon

from .constants_and_params import (
    OUTPUT_DIR,
    SIGNIFICANCE_LEVEL,
    PROBLEMS_TO_TEST,
    AGENTS_TO_TEST,
)


def conduct_wilcoxon_tests():
    for experiment in PROBLEMS_TO_TEST:
        algorithms_best_results = []
        for agent in AGENTS_TO_TEST:
            best_results_for_agent = []
            for filename in os.listdir(path=OUTPUT_DIR):
                regex = rf".*{agent}_{experiment}.*\.csv"
                if re.match(regex, filename):
                    current_df = pd.read_csv(f"{OUTPUT_DIR}/{filename}")

                    best_value = current_df["score"].min()
                    best_results_for_agent.append(best_value)
            algorithms_best_results.append(best_results_for_agent)

        if (
            wilcoxon(algorithms_best_results[0], algorithms_best_results[1]).pvalue
            <= SIGNIFICANCE_LEVEL
        ):
            print(f"STATISTICALLY DIFFERENT for: {experiment}")
        else:
            print(f"statistically the same for: {experiment}")


if __name__ == "__main__":
    conduct_wilcoxon_tests()
