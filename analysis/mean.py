import datetime
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from .constants import MEAN_PLOTS_DIR, OUTPUT_DIR


def plot_and_save_graphs_with_mean_best_results_for_each_iteration():
    experiments = [
        "BaseAgent_ExpandedSchaffer",
        "AgentWithTrust_ExpandedSchaffer",
        "BaseAgent_Griewank",
        "AgentWithTrust_Griewank",
        "BaseAgent_Rastrigin",
        "AgentWithTrust_Rastrigin",
        "BaseAgent_Sphere",
        "AgentWithTrust_Sphere",
    ]
    exp_labels = []
    exp_iter = []
    exp_values = []

    for experiment_name in experiments:
        for filename in os.listdir(OUTPUT_DIR):
            regex = rf".*{experiment_name}.*\.csv"
            if re.match(regex, filename):
                current_df = pd.read_csv(f"{OUTPUT_DIR}/{filename}")

                current_df = current_df.loc[current_df["generation"] <= 1000]

                current_df = current_df.loc[
                    current_df.groupby(["generation"])["score"].idxmax()
                ]
                current_df = current_df[["generation", "score"]]

                exp_labels.extend([experiment_name for _ in range(current_df.shape[0])])
                exp_iter.extend(current_df["generation"].values.tolist())
                exp_values.extend(current_df["score"].values.tolist())

    data = {"exp_label": exp_labels, "iter": exp_iter, "exp_value": exp_values}
    df = pd.DataFrame.from_dict(data)

    for i, exp_name in enumerate(experiments):
        if i % 2 == 0:
            fig, ax = plt.subplots(1, 1)

        current_df = df.loc[df["exp_label"] == exp_name]

        exp_data = []
        iter_labels = []

        for iter_label in range(1, 999):
            iter_labels.append(iter_label)

            iter_exp_values = current_df.loc[current_df["iter"] == iter_label]
            exp_data.append(iter_exp_values["exp_value"].values.mean().tolist())

        ax.plot(iter_labels, exp_data, label=exp_name.split("_")[0])

        if i % 2 == 1:
            fig.legend()
            ax.set_title(
                f"Średnia wartość najlepszego dopasowania w każdej iteracji - {exp_name.split("_")[1]}"
            )
            ax.set_xlabel("Numer iteracji")
            ax.set_ylabel("Średnia wartość najlepszego dopasowania")

            # Plot saving.
            if not os.path.exists(MEAN_PLOTS_DIR):
                os.makedirs(MEAN_PLOTS_DIR)
            now = datetime.datetime.now()
            current_date = (
                f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
            )
            plt.subplots_adjust(
                left=0.2, bottom=0.1, right=0.8, top=0.95, wspace=0.4, hspace=0.4
            )
            fig.set_size_inches(10, 7)
            fig.savefig(
                f"{MEAN_PLOTS_DIR}/{exp_name}_graph_{current_date}.png", dpi=100
            )


if __name__ == "__main__":
    plot_and_save_graphs_with_mean_best_results_for_each_iteration()
