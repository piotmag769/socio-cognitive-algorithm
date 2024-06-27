import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import re


if __name__ == "__main__":

    '''
    > py benchmark_analysis.py

    This script generates plots showing
    the progression of fitness between iterations
    for multiple problems and agent types. 
    '''

    # File loading
    search_path = "./output/50_tests/"
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
    exp_iter = []
    exp_values = []

    number_of_agents = 10
    iteration_interval = 50
    steps_count = None

    for experiment_name in experiments:
        for fname in os.listdir(path=search_path):
            r = ".*(" + experiment_name + "){1}.*\.csv"
            if re.match(r, fname):
                current_df = pd.read_csv(search_path + fname)             

                current_df = current_df.loc[current_df['generation'] <= 1000]
                current_df = current_df.loc[current_df['generation'] % iteration_interval == 0]

                current_df = current_df.loc[current_df.groupby(['generation'])['score'].idxmax()]
                current_df = current_df[['generation', 'score']]
                
                exp_labels.extend([experiment_name for _ in range(current_df.shape[0])])
                exp_iter.extend(current_df['generation'].values.tolist())
                exp_values.extend(current_df['score'].values.tolist())

                if steps_count is None:
                    steps_count = current_df.shape[0]

    data = {'exp_label': exp_labels, 'iter': exp_iter, 'exp_value': exp_values}
    df = pd.DataFrame.from_dict(data)

    # Plot drawing
    '''
    Tweak parameters below when adding new problems or agents!
    '''

    for exp_name in experiments:
        
        current_df = df.loc[df['exp_label'] == exp_name]

        fig, ax = plt.subplots(1, 1)

        exp_data = []
        iter_labels = []

        for i in range(steps_count):
            
            iter_label = iteration_interval * (i+1)
            iter_labels.append(iter_label)
            
            iter_exp_values = current_df.loc[current_df['iter'] == iter_label]
            exp_data.append(iter_exp_values['exp_value'].values.tolist()) 

        ax.boxplot(
            exp_data,
            labels=iter_labels
            )
        ax.set_title(f"{exp_name} - 50 runs")

        # Plot saving
        if not os.path.exists("./graphs/iter"):
            os.makedirs("./graphs/iter")
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
        fig.set_size_inches(10, 7)
        fig.savefig(f"./graphs/iter/{exp_name}_graph_{current_date}.png", dpi=100)