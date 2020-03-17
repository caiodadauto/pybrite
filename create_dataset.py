import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

import pybrite as pb


def interval(s):
    try:
        min_, max_ = map(int, s.split(','))
        return min_, max_
    except:
        raise argparse.ArgumentTypeError("Interval must be min,max")

def create_data(size, path, n_interval, m_interval, offset):
    pb.create_static_dataset(os.path.abspath(path), size, n_interval, m_interval, offset=offset)
    get_stats(path)

def save_figs(data, params):
    for x, y, name in params:
        sns.jointplot(x=x, y=y, data=data, kind="hex", color="k")
        plt.savefig(name)
        plt.close()

def get_stats(path):
    path = Path(path)
    graph_paths = list(path.glob("*.gpickle"))
    print("Generate Stats...", end="\t")
    data_stats = np.zeros((len(graph_paths), 4))
    for i, f in enumerate(graph_paths):
        G = nx.read_gpickle(str(f))
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        avg_degree = n_edges / n_nodes
        assortativity = nx.degree_pearson_correlation_coefficient(G, weight="distance")
        degree_centrality = list(nx.degree_centrality(G).values())
        closeness_centrality = list(nx.closeness_centrality(G, distance="distance").values())
        cov = np.corrcoef(closeness_centrality, degree_centrality)[0,1]
        data_stats[i] = [n_nodes, avg_degree, assortativity, cov]
    df_stats = pd.DataFrame(data_stats, columns=["Number of Nodes", "Average Degree", "Assortativity Coefficient", "Corr. Closeness x Degree"])
    save_figs(df_stats,
              [("Number of Nodes", "Average Degree", path.parent.joinpath("avg_degree_{}.pdf".format(path.name))),
               ("Average Degree", "Assortativity Coefficient", path.parent.joinpath("assortativity_{}.pdf".format(path.name))),
               ("Number of Nodes", "Corr. Closeness x Degree", path.parent.joinpath("coor_{}.pdf".format(path.name)))])
    print("Done.")

def clean_dir(path):
    path= Path(path)
    for p in list(path.glob("*.gpickle")):
        os.remove(p)

def get_last(path):
    path= Path(path)
    sorted_list = sorted(path.glob("*.gpickle"), key=lambda x: int(x.stem))
    if sorted_list:
        return int(sorted_list[-1].stem) + 1
    return 0

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("size", type=int, help="Number of graphs")
    p.add_argument("--path", type=str, default="dataset/", help="Path to save data")
    p.add_argument("--n-interval", type=interval, default=(8,20), help="Interval for number of nodes")
    p.add_argument("--m-interval", type=interval, default=(2,2), help="Interval for connectivity degree")
    p.add_argument("--test", action="store_true", help="Save data for test")
    p.add_argument("--generalization", action="store_true", help="Save data for test in the generalization directory")
    p.add_argument("--overwrite", action="store_true", help="Append to a dataset already existent")
    args = p.parse_args()

    if args.test:
        if args.generalization:
            path = os.path.join(args.path, "test_generalization/")
        else:
            path = os.path.join(args.path, "test_non_generalization/")
    else:
        path = os.path.join(args.path, "train/")

    if not (os.path.isdir(path)):
        os.makedirs(path)
        offset = 0
    else:
        if args.overwrite:
            go = input("The data will be overwrite, do you want to continue? [yN]")
            if go == "y":
                clean_dir(path)
                offset = 0
            else:
                print("Abort the data creation.")
                exit()
        else:
            offset = get_last(path)

    create_data(args.size, path, args.n_interval, args.m_interval, offset)
