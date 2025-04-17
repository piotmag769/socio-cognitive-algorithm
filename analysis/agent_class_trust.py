from collections import defaultdict
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .constants_and_params import (
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
    MULTI_CLASS_PLOTS_DIR,
    STARTING_TRUST,
)

ITERATION_INTERVAL = 50
# BEST_TO_PLOT = 5

# Script Params
data_dir = (
    OUTPUT_DIR + "/2025_4_16_5_13_22"
)  # Make sure that you choose a dir that has experiments with the same agent setup
exp_name = "LABS"  # Title based on Problem, Nr of runs and Agent Combination


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

    # Init trust dict
    trust_dict = {
        current_class: {
            other_class: {generation: 0 for generation in df["generation"].unique()}
            for other_class in class_to_agent_ids.keys()
        }
        for current_class in class_to_agent_ids.keys()
    }
    # Starting trust values
    for current_class in trust_dict.keys():
        for other_class in trust_dict[current_class].keys():
            trust_dict[current_class][other_class][0] = STARTING_TRUST

    # Iteration over each generation
    for generation in df["generation"].unique():
        generation_df = df.loc[df["generation"] == generation]

        # Iteration over each agent_id
        for _, row in generation_df.iterrows():
            agent_id = row["agent_id"]
            trust_string = row["trust"]
            trust_string = [trust for trust in trust_string.split("_")]
            trust_string = trust_string[:-1]
            trust_per_id = {
                int(trust.split(":")[0]): int(trust.split(":")[1])
                for trust in trust_string
            }
            # Add missing keys with STARTING_TRUST value
            for missing_id in agent_id_to_class.keys():
                if missing_id not in trust_per_id.keys():
                    trust_per_id[missing_id] = STARTING_TRUST
            # Sum of trust values per class from this agent_id
            trust_per_class = {}
            for id, trust in trust_per_id.items():
                class_name = agent_id_to_class[id]
                if class_name not in trust_per_class.keys():
                    trust_per_class[class_name] = 0
                trust_per_class[class_name] += int(trust)
            # Filling in trust_dict with trust values per class from position of this agent_id
            for class_name, trust in trust_per_class.items():
                trust_dict[agent_id_to_class[agent_id]][class_name][generation] += trust

        # Normalizing trust values (dividing per agents count x class count)
        for current_class in trust_dict.keys():
            for other_class in trust_dict[current_class].keys():
                trust_dict[current_class][other_class][generation] /= len(
                    class_to_agent_ids[current_class]
                ) * len(class_to_agent_ids[other_class])

    num_classes = len(trust_dict)
    fig, axes = plt.subplots(
        2 * num_classes, 1, figsize=(10, 5 * 2 * num_classes), sharex=True
    )
    # Shared legend.
    lines = []
    labels = []

    color_map = (
        sns.color_palette("Set1", 9)
        + sns.color_palette("Set2", 8)
        + sns.color_palette("Set3", 7)
    )
    class_colors = {
        class_name: color_map[i]
        for i, class_name in enumerate(class_to_agent_ids.keys())
    }

    # Trust FROM
    trust_from_other_agents = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for class_from, trusts_to in trust_dict.items():
        for class_to, gen_trusts in trusts_to.items():
            for gen, trust in gen_trusts.items():
                trust_from_other_agents[class_to][class_from][gen] += trust

    for idx, cls in enumerate(class_to_agent_ids.keys()):
        ax_to = axes[2 * idx]
        ax_from = axes[2 * idx + 1]

        # Outgoing Trust
        for class_to, gen_trusts in trust_dict[cls].items():
            if cls == class_to:
                continue
            generations = sorted(gen_trusts.keys())
            trust_scores = [100 - gen_trusts[generation] for generation in generations]
            ax_to.plot(
                generations,
                trust_scores,
                label=f"Trust to {class_to}",
                color=class_colors[class_to],
            )

        ax_to.set_title(f"Trust Given by {cls}")
        ax_to.grid(True)

        # Incoming Trust
        for class_from, gen_trusts in trust_from_other_agents[cls].items():
            if cls == class_to:
                continue

            generations = sorted(gen_trusts.keys())
            trust_scores = [100 - gen_trusts[generation] for generation in generations]
            (line,) = ax_from.plot(
                generations,
                trust_scores,
                label=f"Trust from {class_from}",
                color=class_colors[class_from],
            )
            if class_from not in labels:
                lines.append(line)
                labels.append(class_from)
        ax_from.set_title(f"Trust Received by {cls}")
        ax_from.grid(True)

    axes[-1].set_xlabel("Generation")

    # Adding a shared legend outside the plot
    fig.legend(lines, labels, loc="center right", borderaxespad=0.1)
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
        f"{MULTI_CLASS_PLOTS_DIR}/TRUST_TO_{exp_name}__{current_date}.png",
        dpi=100,
    )


if __name__ == "__main__":
    plot_and_save_average_agent_class_trust_in_training()
