import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import re


if __name__ == "__main__":

    '''
    > py benchmark_analysis.py

    This script generates a compliation of plots 
    describing a benchmark comparing results gathered 
    from different implementations of Agent behaviors 
    within the Island evolution algorithm.
    '''

    # File loading
    search_path = "./output/"
    '''
    Experiment names order matters!!!
    It's used later for plotting order.
    Group the names by problem type and have
    the order of agents consistent between
    the problem types.
    '''
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
        for fname in os.listdir(path=search_path):
            r = ".*(" + experiment_name + "){1}.*\.csv"
            if re.match(r, fname):
                with open(search_path + fname, 'rb') as f:
                    try: 
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                    except OSError:
                        f.seek(0)
                    last_line = f.readline().decode()
                    experiment_value = float(last_line.split(',')[-1])
                    exp_labels.append(experiment_name)
                    exp_values.append(experiment_value)

    data = {'exp_label': exp_labels, 'exp_value': exp_values}
    df = pd.DataFrame.from_dict(data)

    # Plot drawing
    '''
    Tweak parameters below when adding new problems or agents!
    '''
    number_of_problems = 4
    number_of_agent_types = 2
    fig, axes = plt.subplots(number_of_problems, 1)

    for i in range(number_of_problems):
        
        ax = axes[i]
        problem_label = experiments[i * number_of_agent_types].split('_')[-1]
        
        exp_data = []
        agent_labels = []
        for j in range(number_of_agent_types):
            exp_label = experiments[i*number_of_agent_types + j]
            agent_labels.append(exp_label.split('_')[0])
            exp_values = df.loc[df['exp_label'] == exp_label]
            exp_data.append(exp_values['exp_value'].values.tolist()) 

        ax.boxplot(
            exp_data,
            labels=agent_labels
            )
        ax.set_title(problem_label)


    # Plot saving
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")
    now = datetime.datetime.now()
    current_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    plt.subplots_adjust(left=0.2,
                    bottom=0.05, 
                    right=0.8, 
                    top=0.95, 
                    wspace=0.4, 
                    hspace=0.4)
    fig.set_size_inches(3 * number_of_agent_types, 4 * number_of_problems)
    fig.savefig(f"./graphs/compilation_graph_{current_date}.png", dpi=100)