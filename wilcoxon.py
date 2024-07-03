import pandas as pd
import os
import pandas as pd
import re

from scipy.stats import wilcoxon


if __name__ == "__main__":
    search_path = "./output/"

    agents = ["BaseAgent", "AgentWithTrust"]
    experiments = [
        "ExpandedSchaffer",
        "Griewank",
        "Rastrigin",
        "Sphere",
    ]

    for experiment in experiments:
        test_args = []
        for agent in agents:
            best_results_for_agent = []
            for fname in os.listdir(path=search_path):
                r = ".*(" + agent + "_" + experiment + r"){1}.*\.csv"
                if re.match(r, fname):
                    current_df = pd.read_csv(search_path + fname)

                    best_value = current_df["score"].max()
                    best_results_for_agent.append(best_value)
            test_args.append(best_results_for_agent)
        print(experiment)
        print(wilcoxon(test_args[0], test_args[1]))
