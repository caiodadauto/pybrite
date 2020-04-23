import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

import pybrite as pb


def save_figs(data, params):
    for x, y, name in params:
        fig = plt.figure(dpi=300)
        ax = fig.subplots(1, 1, sharey=False)
        # min_x = data[x].values.min() * 1.1 if data[x].values.min() < 0 else 0
        # max_x = data[x].values.max() * 1.1 if data[x].values.max() > 0 else 0
        # min_y = data[y].values.min() * 1.1 if data[y].values.min() < 0 else 0
        # max_y = data[y].values.max() * 1.1 if data[y].values.max() > 0 else 0
        # sns.jointplot(x=x, y=y, data=data, kind="hex", color="k", xlim=(min_x, max_x), ylim=(min_y, max_y))
        data.plot.scatter(x, y, ax=ax)

        # ax.set_xticks(np.arange(min_x, max_x, .25))
        ax.set_yticks(np.arange(min_y, max_y, .25))
        # ax.yaxis.grid(True)
        fig.tight_layout()

        plt.savefig(name, transparent=True)
        fig.clear()
        plt.close()

def get_stats(path, path_stats):
    sns.set_style("ticks")
    path_stats = Path(path_stats)
    path = Path(path)
    graph_paths = list(path.glob("*.gpickle"))
    print("Generate Stats...", end="\t")
    data_stats = np.zeros((len(graph_paths), ), dtype=[
        ("Number of Nodes", "i4"),
        ("Average Degree", "f8"),
        ("Maximum Degree", "f8"),
        ("Normalized Maximum Degree", "f8"),
        ("High-Degree Ratio", "f8"),
        ("Assortativity Coefficient", "f8"),
        ("Corr. Closeness x Degree", "f8"),
        ("Class", "U2")
    ])
    for i, f in enumerate(graph_paths):
        G = nx.read_gpickle(str(f))
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        closeness_centrality = np.array(list(nx.closeness_centrality(G, distance="distance").values()))
        assortativity = nx.degree_pearson_correlation_coefficient(G, weight="distance")
        degree = np.array(list(dict(nx.degree(G)).values()))
        cov = np.corrcoef(closeness_centrality, degree)[0,1]
        avg_degree = (2 * n_edges) / n_nodes
        max_degree = degree.max()
        max_degree_norm = max_degree / n_nodes
        R = (degree > avg_degree).sum() / n_nodes
        top_class = "L" if max_degree_norm < 0.4 and avg_degree < 3 else "NL"

        data_stats[i] = (n_nodes, avg_degree, max_degree, max_degree_norm, R, assortativity, cov, top_class)
    df_stats = pd.DataFrame(data_stats)

    save_figs(df_stats, [
        ( "Number of Nodes", "Average Degree", path_stats.joinpath("avg_degree_{}.pdf".format(path.name)) ),
        ( "Average Degree", "Assortativity Coefficient", path_stats.joinpath("assortativity_{}.pdf".format(path.name)) ),
        ( "Number of Nodes", "Corr. Closeness x Degree", path_stats.joinpath("coor_{}.pdf".format(path.name)) )
    ])

    save_figs(df_stats, [
        ( "Normalized Maximum Degree", "Average Degree", path_stats.joinpath("norm_avg_{}.pdf".format(path.name)) ),
    ])

    non_ladder = df_stats.loc[df_stats["Class"] == "NL"]
    if not non_ladder.empty:
        save_figs(non_ladder, [
            ( "High-Degree Ratio", "Maximum Degree", path_stats.joinpath("r_{}.pdf".format(path.name)) )
        ])
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, default="dataset/", help="Path to datatset")
    p.add_argument("--test", action="store_true", help="Save data for test")
    p.add_argument("--generalization", action="store_true", help="Save data for test in the generalization directory")
    args = p.parse_args()

    if args.test:
        if args.generalization:
            path = os.path.join(args.path, "test_generalization/")
        else:
            path = os.path.join(args.path, "test_non_generalization/")
    else:
        path = os.path.join(args.path, "train/")
    path_stats = os.path.join(args.path, "stats")

    if not os.path.isdir(path_stats):
        os.makedirs(path_stats)
    if not (os.path.isdir(path)):
        os.makedirs(path)

    get_stats(path, path_stats)
