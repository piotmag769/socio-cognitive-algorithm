import os
import re

import pandas as pd

from scipy.stats import wilcoxon

from .constants import OUTPUT_DIR, SIGNIFICANCE_LEVEL


def conduct_wilcoxon_tests():
    agents = [
        "BaseAgent",
        "AgentWithTrust",
    ]
    experiments = [
        "ExpandedSchaffer",
        "Griewank",
        "Rastrigin",
        "Sphere",
    ]

    for experiment in experiments:
        algorithms_best_results = []
        for agent in agents:
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
