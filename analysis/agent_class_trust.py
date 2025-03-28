import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .constants_and_params import (
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
    MULTI_CLASS_PLOTS_DIR,
    GENERATIONS_PER_SWAP,
    STARTING_TRUST,
)

ITERATION_INTERVAL = 50
# BEST_TO_PLOT = 5

# Script Params
data_dir = (
    OUTPUT_DIR + "/2025_3_28_21_19_32_MIGRATION"
)  # Make sure that you choose a dir that has experiments with the same agent setup
exp_name = "Griewank_Migration"  # Title based on Problem, Nr of runs and Agent Combination


def plot_and_save_average_agent_class_trust_in_training():

    ''' Extracting data into a single dataframe '''
    dataframes = []
    for filename in os.listdir(path=data_dir):
        current_df = pd.read_csv(f"{data_dir}/{filename}")
        current_df = current_df.loc[current_df["generation"] <= NUMBER_OF_ITERATIONS]
        current_df = current_df.loc[current_df["generation"] % ITERATION_INTERVAL == 0]
        dataframes.append(current_df)
    df = pd.concat(dataframes, ignore_index=False)

    ''' Data cleaning and processing '''
    first_generation_df = current_df.loc[current_df["generation"] == ITERATION_INTERVAL]
    unique_ids_and_classes = first_generation_df[["agent_id", "class"]].drop_duplicates()
    agent_id_to_class = unique_ids_and_classes.set_index("agent_id")["class"].to_dict()
    class_to_agent_ids = unique_ids_and_classes.groupby("class")["agent_id"].apply(list).to_dict()
    
    # Init trust dict
    trust_dict = {
        current_class: {
            other_class: {
                generation: 0 for generation in df["generation"].unique()
            }
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
            trust_per_id = { int(trust.split(":")[0]): int(trust.split(":")[1]) for trust in trust_string }
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
                trust_dict[current_class][other_class][generation] /= len(class_to_agent_ids[current_class]) * len(class_to_agent_ids[other_class])

    ''' Plotting trust values per generation for each agent class '''
    num_classes = len(trust_dict.keys())
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes), sharex=True)

    if num_classes == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one class

    for idx, (current_class, other_classes) in enumerate(trust_dict.items()):
        ax = axes[idx]
        # max_trust_value = max(
        #     trust_value
        #     for trust_values in other_classes.values()
        #     for trust_value in trust_values.values()
        # )
        max_trust_value = 20  # Set a fixed max trust value for all plots
        min_trust_value = min(
            trust_value
            for trust_values in other_classes.values()
            for trust_value in trust_values.values()
        )

        for other_class, trust_values in other_classes.items():
            generations = sorted(trust_values.keys())
            trust_scores = [trust_values[generation] for generation in generations]
            ax.plot(generations, trust_scores, label=f"Trust in {other_class}")

        ax.set_title(f"Trust Values for Class {current_class}")
        ax.set_ylim(min_trust_value - 5, max_trust_value + 5)  # Adjust ylim dynamically
        ax.set_ylabel("Trust")
        ax.grid(visible=True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Generation")
    plt.tight_layout(pad=5.0)  # Add more vertical padding between plots
    plt.show()

    ''' Plot config '''
    
    # Plot saving.
    os.makedirs(MULTI_CLASS_PLOTS_DIR, exist_ok=True)

    now = datetime.datetime.now()
    current_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    plt.subplots_adjust(
        left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.4, hspace=0.8
    )  # Adjust hspace for better spacing

    fig.savefig(f"{MULTI_CLASS_PLOTS_DIR}/{exp_name}_graph_{current_date}.png", dpi=100)


if __name__ == "__main__":
    plot_and_save_average_agent_class_trust_in_training()
