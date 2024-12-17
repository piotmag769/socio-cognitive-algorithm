import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .constants_and_params import (
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
    MULTI_CLASS_PLOTS_DIR,
)

ITERATION_INTERVAL = 100
BEST_TO_PLOT = 5

# Script Params
data_dir = (
    OUTPUT_DIR + "/2024_12_2_4_14_36"
)  # Make sure that you choose a dir that has experiments with the same agent setup
exp_name = "Schaffer_MIGRATION_100var_5run_AllDifferent_LONG"  # Title based on Problem, Nr of runs and Agent Combination


def plot_and_save_average_agent_class_performance_in_training():
    dataframes = []

    for filename in os.listdir(path=data_dir):
        current_df = pd.read_csv(f"{data_dir}/{filename}")
        current_df = current_df.loc[current_df["generation"] <= NUMBER_OF_ITERATIONS]
        current_df = current_df.loc[current_df["generation"] % ITERATION_INTERVAL == 0]
        dataframes.append(current_df)

    df = pd.concat(dataframes, ignore_index=False)

    # Plot drawing.
    # Tweak parameters below when adding new problems or agents!
    fig, ax = plt.subplots(1, 1)

    steps_count = df["generation"].max() // ITERATION_INTERVAL
    iter_labels = np.arange(1, df["generation"].max() + 1, ITERATION_INTERVAL)

    mean_std_final_agent_type_datas = []
    for agent_type in df["class"].unique():
        mean_data = []
        std_data = []

        for i in range(steps_count):
            iter_label = ITERATION_INTERVAL * (i + 1)

            iter_exp_values = df.loc[
                ((df["generation"] == iter_label) & (df["class"] == agent_type))
            ]
            mean_data.append(iter_exp_values["score"].values.mean(axis=0))
            std_data.append(iter_exp_values["score"].values.std(axis=0, ddof=1))

        mean_std_final_agent_type_datas.append(
            (np.array(mean_data), np.array(std_data), mean_data[-1], agent_type)
        )

    mean_std_final_agent_type_datas.sort(key=lambda x: x[2])

    for i in range(BEST_TO_PLOT):
        mean_data, std_data, final_y, agent_type = mean_std_final_agent_type_datas[i]
        ax.plot(iter_labels, mean_data, label=agent_type)
        ax.fill_between(
            iter_labels, mean_data - std_data, mean_data + std_data, alpha=0.2
        )

        plt.annotate(
            f"{final_y:.2f}",  # Annotate with the final value (formatted to 2 decimals)
            (iter_labels[-1], final_y),  # The point to annotate
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
            textcoords="offset points",  # Position the text relative to the point
            xytext=(25, -15 * i),  # Offset the text by (x, y) pixels
            arrowprops=dict(arrowstyle="-", color="gray"),
            fontsize=10,
            color="black",
        )

    ax.set_title(exp_name)
    ax.legend(loc="upper right")

    # Plot saving.
    os.makedirs(MULTI_CLASS_PLOTS_DIR, exist_ok=True)

    now = datetime.datetime.now()
    current_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    plt.subplots_adjust(
        left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.4, hspace=0.4
    )
    fig.set_size_inches(10, 7)
    fig.savefig(f"{MULTI_CLASS_PLOTS_DIR}/{exp_name}_graph_{current_date}.png", dpi=100)


if __name__ == "__main__":
    plot_and_save_average_agent_class_performance_in_training()
