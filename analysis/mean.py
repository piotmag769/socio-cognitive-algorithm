import datetime
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from .constants_and_params import (
    MEAN_PLOTS_DIR,
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
    EXPERIMENTS,
)


def plot_and_save_graphs_with_mean_best_results_for_each_iteration():
    exp_labels = []
    exp_iter = []
    exp_values = []

    for experiment_name in EXPERIMENTS:
        for filename in os.listdir(OUTPUT_DIR):
            regex = rf".*{experiment_name}.*\.csv"
            if re.match(regex, filename):
                current_df = pd.read_csv(f"{OUTPUT_DIR}/{filename}")

                current_df = current_df.loc[
                    current_df["generation"] <= NUMBER_OF_ITERATIONS
                ]

                current_df = current_df.loc[
                    current_df.groupby(["generation"])["score"].idxmax()
                ]
                current_df = current_df[["generation", "score"]]

                exp_labels.extend([experiment_name for _ in range(current_df.shape[0])])
                exp_iter.extend(current_df["generation"].values.tolist())
                exp_values.extend(current_df["score"].values.tolist())

    data = {"exp_label": exp_labels, "iter": exp_iter, "exp_value": exp_values}
    df = pd.DataFrame.from_dict(data)

    fig, ax = plt.subplots(1, 1)
    for i, exp_name in enumerate(EXPERIMENTS):

        current_df = df.loc[df["exp_label"] == exp_name]

        exp_data = []
        iter_labels = []

        for iter_label in range(1, NUMBER_OF_ITERATIONS + 1):
            iter_labels.append(iter_label)

            iter_exp_values = current_df.loc[current_df["iter"] == iter_label]
            exp_data.append(iter_exp_values["exp_value"].values.mean().tolist())

        if exp_data[-1] <= 2400:
            ax.plot(iter_labels, exp_data, label=exp_name)
            fig.legend()
    ax.set_title(f"Średnia wartość najlepszego dopasowania w każdej iteracji")
    ax.set_xlabel("Numer iteracji")
    ax.set_ylabel("Średnia wartość najlepszego dopasowania")

    # Plot saving.
    os.makedirs(MEAN_PLOTS_DIR, exist_ok=True)
    now = datetime.datetime.now()
    current_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    plt.subplots_adjust(
        left=0.2, bottom=0.1, right=0.8, top=0.95, wspace=0.4, hspace=0.4
    )
    fig.set_size_inches(10, 7)
    fig.savefig(f"{MEAN_PLOTS_DIR}/{exp_name}_graph_{current_date}.png", dpi=100)


if __name__ == "__main__":
    plot_and_save_graphs_with_mean_best_results_for_each_iteration()
