from collections import defaultdict
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants_and_params import (
    ITERATION_INTERVAL,
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
    MULTI_CLASS_PLOTS_DIR,
)

# BEST_TO_PLOT = 5

exp_name = "LABS"  # Title based on Problem, Nr of runs and Agent Combination
data_dir = f"{OUTPUT_DIR}/new/{exp_name}"  # Make sure that you choose a dir that has experiments with the same agent setup


def plot_and_save_average_agent_class_trust_in_training():
    """Extracting data into a single dataframe"""
    dataframes = []
    for filename in os.listdir(path=data_dir):
        current_df = pd.read_csv(f"{data_dir}/{filename}")
        current_df = current_df.loc[current_df["generation"] <= NUMBER_OF_ITERATIONS]
        current_df = current_df.loc[current_df["generation"] % ITERATION_INTERVAL == 0]
        dataframes.append(current_df)
    df = pd.concat(dataframes, ignore_index=False)

    """ Data cleaning and processing """
    first_generation_df = current_df.loc[current_df["generation"] == ITERATION_INTERVAL]
    unique_ids_and_classes = first_generation_df[
        ["agent_id", "class"]
    ].drop_duplicates()
    agent_id_to_class = unique_ids_and_classes.set_index("agent_id")["class"].to_dict()
    class_to_agent_ids = (
        unique_ids_and_classes.groupby("class")["agent_id"].apply(list).to_dict()
    )

    # Initialization for trust collection
    trust_given = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    trust_received = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Gather trust data
    for generation, generation_df in df.groupby("generation"):
        for _, row in generation_df.iterrows():
            agent_id = row["agent_id"]
            class_from = agent_id_to_class[agent_id]
            trust_entries = [tr.split(":") for tr in row["trust"][:-1].split("_")]
            trust_per_id = {int(agent): int(trust) for agent, trust in trust_entries}
            for agent, trust in trust_per_id.items():
                class_to = agent_id_to_class[agent]
                trust_given[class_from][class_to][generation].append(trust)
                trust_received[class_to][class_from][generation].append(trust)

    # Calculate mean and standard deviation
    def calculate_stats(trust_data):
        return {
            class_from: {
                class_to: {
                    generation: (np.mean(trusts), np.std(trusts, ddod=1))
                    for generation, trusts in gen_trusts.items()
                }
                for class_to, gen_trusts in trusts_to.items()
            }
            for class_from, trusts_to in trust_data.items()
        }

    trust_given = calculate_stats(trust_given)
    trust_received = calculate_stats(trust_received)

    num_classes = len(trust_given)
    fig, axes = plt.subplots(
        2 * num_classes, 1, figsize=(10, 5 * 2 * num_classes), sharex=True
    )
    # Shared legend.
    lines = []
    labels = []

    color_map = (
        sns.color_palette("Set1")
        + sns.color_palette("Set2")
        + sns.color_palette("Set3")
    )
    class_colors = {
        class_name: color_map[i]
        for i, class_name in enumerate(class_to_agent_ids.keys())
    }

    for idx, cls in enumerate(class_to_agent_ids.keys()):
        ax_to = axes[2 * idx]
        ax_from = axes[2 * idx + 1]

        # Outgoing Trust
        for class_to, gen_trusts in trust_given[cls].items():
            if cls == class_to:
                continue
            generations = sorted(gen_trusts.keys())
            trust_scores = np.array(
                [-gen_trusts[generation][0] for generation in generations]
            )
            stds = np.array([gen_trusts[generation][1] for generation in generations])
            ax_to.plot(
                generations,
                trust_scores,
                color=class_colors[class_to],
            )
            ax_to.fill_between(
                generations,
                trust_scores - stds,
                trust_scores + stds,
                alpha=0.2,
                color=class_colors[class_to],
            )
            ax_to.annotate(
                class_to,  # Annotate with the final value (formatted to 2 decimals)
                (generations[-1], trust_scores[-1]),  # The point to annotate
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
                textcoords="offset points",  # Position the text relative to the point
                xytext=(25, -15 * 2),  # Offset the text by (x, y) pixels
                arrowprops=dict(arrowstyle="-", color="gray"),
                fontsize=10,
                color="black",
            )

        ax_to.set_title(f"Trust Given by {cls}")
        ax_to.grid(True)

        # Incoming Trust
        for class_from, gen_trusts in trust_received[cls].items():
            if cls == class_from:
                continue

            generations = sorted(gen_trusts.keys())
            trust_scores = np.array(
                [-gen_trusts[generation][0] for generation in generations]
            )
            stds = np.array([gen_trusts[generation][1] for generation in generations])
            (line,) = ax_from.plot(
                generations,
                trust_scores,
                color=class_colors[class_from],
            )
            ax_from.fill_between(
                generations,
                trust_scores - stds,
                trust_scores + stds,
                alpha=0.2,
                color=class_colors[class_from],
            )
            ax_from.annotate(
                class_from,  # Annotate with the final value (formatted to 2 decimals)
                (generations[-1], trust_scores[-1]),  # The point to annotate
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
                textcoords="offset points",  # Position the text relative to the point
                xytext=(25, -15 * 2),  # Offset the text by (x, y) pixels
                arrowprops=dict(arrowstyle="-", color="gray"),
                fontsize=10,
                color="black",
            )
            if class_from not in labels:
                lines.append(line)
                labels.append(class_from)
        ax_from.set_title(f"Trust Received by {cls}")
        ax_from.grid(True)

    axes[-1].set_xlabel("Generation")

    # Adding a shared legend outside the plot
    fig.legend(lines, labels, loc="center left", borderaxespad=0.1)
    # Adjust layout and save the figure
    plt.tight_layout()

    plt.subplots_adjust(
        left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.4, hspace=0.8
    )  # Adjust hspace for better spacing

    now = datetime.datetime.now()
    current_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    os.makedirs(MULTI_CLASS_PLOTS_DIR, exist_ok=True)

    # Plot saving.
    fig.savefig(
        f"{MULTI_CLASS_PLOTS_DIR}/TRUST_{exp_name}__{current_date}.png",
        dpi=100,
    )


if __name__ == "__main__":
    plot_and_save_average_agent_class_trust_in_training()
