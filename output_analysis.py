import sys
import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd

# py output_analysis.py "output_file_path_1" "output_file_path_2"

if __name__ == "__main__":

    # Loading the files
    df1 = pd.read_csv(sys.argv[1])
    df2 = pd.read_csv(sys.argv[2])

    # Plot creation
    fig = plt.figure(1)

    plt.subplot(1,1,1, title="Best Score Per Generation", xlabel="Generation Number", ylabel="Score")
    plt.plot(df1["generation"].unique(), df1.groupby(['generation'])['score'].max(), "tab:green", label=sys.argv[1])
    plt.plot(df2["generation"].unique(), df2.groupby(['generation'])['score'].max(), "tab:purple", label=sys.argv[2])
    plt.legend()

    # Saving of the plot
    if not os.path.exists("./graphs"):
        os.makedirs("./graphs")
    now = datetime.datetime.now()
    current_date = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    fig.set_size_inches(18, 10)
    fig.savefig(f'./graphs/graph_{current_date}.png', dpi=100)