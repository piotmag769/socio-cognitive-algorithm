import os
import datetime
import re

import matplotlib.pyplot as plt
import pandas as pd

from .constants import OUTPUT_DIR, PLOTS_DIR


def plot_and_save_summary_box_and_whiskers_comparision_graph_with_final_results():
    # Experiment names order matters!!!
    # It's used later for plotting order.
    # Group the names by problem type and have
    # the order of agents consistent between
    # the problem types.
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
    exp_values = []

    for experiment_name in experiments:
        for filename in os.listdir(OUTPUT_DIR):
            regex = rf".*{experiment_name}.*\.csv"
            if re.match(regex, filename):
                df = pd.read_csv(f"{OUTPUT_DIR}/{filename}")

                experiment_value = df["score"].min()
                exp_values.append(experiment_value)

                exp_labels.append(experiment_name)

    data = {"exp_label": exp_labels, "exp_value": exp_values}
    df = pd.DataFrame.from_dict(data)

    # Plot drawing.
    # Tweak parameters below when adding new problems or agents!
    number_of_problems = 4
    number_of_agent_types = 2
    fig, axes = plt.subplots(number_of_problems, 1)

    for i in range(number_of_problems):
        ax = axes[i]
        problem_label = experiments[i * number_of_agent_types].split("_")[-1]

        exp_data = []
        agent_labels = []
        for j in range(number_of_agent_types):
            exp_label = experiments[i * number_of_agent_types + j]
            agent_labels.append(exp_label.split("_")[0])
            exp_values = df.loc[df["exp_label"] == exp_label]
            exp_data.append(exp_values["exp_value"].values.tolist())

        ax.boxplot(exp_data, labels=agent_labels)
        ax.set_title(problem_label)

    # Plot saving.
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    now = datetime.datetime.now()
    current_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    plt.subplots_adjust(
        left=0.2, bottom=0.05, right=0.8, top=0.95, wspace=0.4, hspace=0.4
    )
    fig.set_size_inches(3 * number_of_agent_types, 4 * number_of_problems)
    fig.savefig(f"{PLOTS_DIR}summary_graph_{current_date}.png", dpi=100)


if __name__ == "__main__":
    plot_and_save_summary_box_and_whiskers_comparision_graph_with_final_results()
