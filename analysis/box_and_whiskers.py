import datetime
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from .constants_and_params import (
    EXPERIMENTS,
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
    BOX_AND_WHISKERS_PLOTS_DIR,
)

ITERATION_INTERVAL = 50


def plot_and_save_box_and_whiskers_graphs_with_best_results_for_some_iterations():
    # limit on the fitness axis
    exp_limits = [
        [44.5, 48],
        [44.5, 48],
        [1300, 2100],
        [1300, 2100],
        [1150, 1500],
        [1150, 1500],
        [350, 600],
        [350, 600],
    ]

    exp_labels = []
    exp_iter = []
    exp_values = []

    steps_count = None

    for experiment_name in EXPERIMENTS:
        for filename in os.listdir(path=OUTPUT_DIR):
            regex = rf".*{experiment_name}.*\.csv"
            if re.match(regex, filename):
                current_df = pd.read_csv(f"{OUTPUT_DIR}/{filename}")

                current_df = current_df.loc[
                    current_df["generation"] <= NUMBER_OF_ITERATIONS
                ]
                current_df = current_df.loc[
                    current_df["generation"] % ITERATION_INTERVAL == 0
                ]

                current_df = current_df.loc[
                    current_df.groupby(["generation"])["score"].idxmax()
                ]
                current_df = current_df[["generation", "score"]]

                exp_labels.extend([experiment_name for _ in range(current_df.shape[0])])
                exp_iter.extend(current_df["generation"].values.tolist())
                exp_values.extend(current_df["score"].values.tolist())

                if steps_count is None:
                    steps_count = current_df.shape[0]

    data = {"exp_label": exp_labels, "iter": exp_iter, "exp_value": exp_values}
    df = pd.DataFrame.from_dict(data)

    # Plot drawing.
    # Tweak parameters below when adding new problems or agents!
    for i, exp_name in enumerate(EXPERIMENTS):
        current_df = df.loc[df["exp_label"] == exp_name]

        fig, ax = plt.subplots(1, 1)

        exp_data = []
        iter_labels = []

        for j in range(steps_count):
            iter_label = ITERATION_INTERVAL * (j + 1)
            iter_labels.append(iter_label)

            iter_exp_values = current_df.loc[current_df["iter"] == iter_label]
            exp_data.append(iter_exp_values["exp_value"].values.tolist())

        ax.boxplot(exp_data, labels=iter_labels)
        ax.set_title(exp_name)
        print(f"{i + 1}/{len(exp_limits)}")
        ax.set_ylim(exp_limits[i])

        # Plot saving.
        os.makedirs(BOX_AND_WHISKERS_PLOTS_DIR, exist_ok=True)

        now = datetime.datetime.now()
        current_date = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        )
        plt.subplots_adjust(
            left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.4, hspace=0.4
        )
        fig.set_size_inches(10, 7)
        fig.savefig(
            f"{BOX_AND_WHISKERS_PLOTS_DIR}/{exp_name}_graph_{current_date}.png", dpi=100
        )


if __name__ == "__main__":
    plot_and_save_box_and_whiskers_graphs_with_best_results_for_some_iterations()
