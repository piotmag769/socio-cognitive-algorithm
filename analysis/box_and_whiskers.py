import datetime
import os

import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd

from .constants_and_params import (
    ITERATION_INTERVAL,
    NUMBER_OF_ITERATIONS,
    OUTPUT_DIR,
    BOX_AND_WHISKERS_PLOTS_DIR,
    PROBLEMS_TO_TEST,
)


def plot_and_save_box_and_whiskers_graphs_with_best_results_for_some_iterations():
    for problem in PROBLEMS_TO_TEST:
        data = {}
        final_scores = {}
        problem_dir = f"{OUTPUT_DIR}/{problem.name()}"
        for root, dirs, _ in os.walk(problem_dir):
            if len(dirs) == 0:
                d, f = get_data(root, problem.name())
                data[str(root)] = d
                final_scores[str(root)] = f
        draw_plot(final_scores, data, problem.name())


def path_to_label(path):
    path = path.strip("./")
    parts = path.split("/")

    scm = parts[3]
    strategy = parts[4] if "starting_trust=" in path else ""
    trust = parts[5].split("=")[1] if "starting_trust=" in path else ""

    scm = {
        "RepelWorstGravity": "RW",
        "RepelWorstGravityMultistep": "RWM",
        "FollowBestDistinctGA": "FBD",
        "FollowBestGA": "FB",
    }.get(scm, None)

    label = "IGA (bazowy)"
    if strategy and scm:
        label = f"IGA + SCM ({scm}) + T"
    elif strategy:
        label = f"IGA + T"
    elif scm:
        label = f"IGA + SCM ({scm})"

    if strategy:
        if strategy == "Global":
            label += " | Globalne zaufanie"
        elif strategy == "Local":
            label += " | Lokalne zaufanie"
        else:
            assert False
    if trust:
        label += f" | z\u2080={trust}"

    return label


def get_data(dir, problem):
    x = -1 if problem == "Knapsack" else 1
    final_scores = []
    data = []
    for filename in os.listdir(path=dir):
        current_df = pd.read_csv(f"{dir}/{filename}")
        best_value = current_df["score"].min()
        final_scores.append(x * best_value)

        current_df = current_df.loc[current_df["generation"] <= NUMBER_OF_ITERATIONS]
        current_df = current_df.loc[
            ((current_df["generation"] - 1) % ITERATION_INTERVAL == 0)
            | (current_df["generation"] == 998)
        ]

        current_df = current_df.loc[
            current_df.groupby(["generation"])["score"].idxmin()
        ]
        current_df = current_df[["generation", "score"]]
        current_df["score"] *= x
        data.append(current_df)
    return data, final_scores


def draw_plot(final_scores, data, problem):
    os.makedirs(BOX_AND_WHISKERS_PLOTS_DIR, exist_ok=True)

    # Sortowanie wg mediany wartości (opcjonalne)
    sorted_items = sorted(final_scores.items(), key=lambda x: np.median(x[1]))
    labels, scores = zip(*[(path_to_label(k), v) for k, v in sorted_items])

    fig, ax = plt.subplots(
        figsize=(13, max(6, len(scores) * 0.3))
    )  # Dopasuj rozmiar wykresu

    # Wykres poziomy (horizontal boxplot)
    box = ax.boxplot(scores, vert=False, patch_artist=True)

    for median in box["medians"]:
        median.set_color("orange")  # Możesz też spróbować: 'darkred'
        median.set_linewidth(2.5)

    # Niestandardowe kolory pudełek
    colors = plt.cm.viridis(np.linspace(0, 1, len(scores)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Ustaw y-ticki i podłącz etykiety
    yticks_pos = range(1, len(labels) + 1)
    ax.set_yticks(yticks_pos)
    text_labels = []
    for label in labels:
        text_labels.append(label)

    ax.set_yticklabels(text_labels, fontsize=11)

    # Wyróżnij "IGA (bazowy)" => znajdź tick i rzuć na niego fontweight=bold
    for tick_label in ax.yaxis.get_ticklabels():
        if tick_label.get_text().strip() == "IGA (bazowy)":
            tick_label.set_fontweight("bold")

    ax.set_title(
        f"Boxplot wyników końcowych - {problem if problem != "Symmetric TSP" else "STSP"}",
        fontsize=20,
    )
    ax.set_xlabel("Końcowe wyniki algorytmów (10 powtórzeń)", fontsize=16)

    baseline_idx = labels.index("IGA (bazowy)")
    baseline_median = np.median(scores[baseline_idx])
    ax.axvline(baseline_median, color="red", linestyle="--", linewidth=1.2, alpha=0.6)

    ax.grid(axis="x", linestyle="--", alpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    fig.savefig(f"{BOX_AND_WHISKERS_PLOTS_DIR}/{problem}_boxplots.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_and_save_box_and_whiskers_graphs_with_best_results_for_some_iterations()
