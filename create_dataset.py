import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

import pytop as pt


def interval(s):
    try:
        min_, max_ = map(int, s.split(','))
        return min_, max_
    except:
        raise argparse.ArgumentTypeError("Interval must be min,max")

def save_figs(data, params):
    sns.set_style("ticks")
    for x, y, name in params:
        fig = plt.figure(dpi=300)
        ax = fig.subplots(1, 1, sharey=False)
        min_x = np.floor(data[x].values.min() * 1.1) if data[x].values.min() < 0 else 0
        min_y = np.floor(data[y].values.min() * 1.1) if data[y].values.min() < 0 else 0
        max_x = np.ceil(data[x].values.max() * 1.1) if data[x].values.max() > 0 else 0
        max_y = np.ceil(data[y].values.max() * 1.1) if data[y].values.max() > 0 else 0
        sns.scatterplot(x=x, y=y, data=data, hue="Topology Class", style="Generator Tool")

        ax.set_xticks(np.arange(min_x, max_x, (max_x - min_x) / 5))
        ax.set_yticks(np.arange(min_y, max_y, (max_y - min_y) / 5))
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        fig.tight_layout()

        plt.savefig(name, transparent=True)
        fig.clear()
        plt.close()

def get_stats(path, path_stats):
    path_stats = Path(path_stats)
    path = Path(path)
    graph_paths = list(path.glob("*.gpickle"))
    print("Generate Stats...", end="\t")
    star_hs = []
    data_stats = np.zeros((len(graph_paths), ), dtype=[
        ("Number of Nodes", "i4"),
        ("Average Degree", "f8"),
        ("Maximum Degree", "f8"),
        ("Normalized Maximum Degree", "f8"),
        ("High-Degree Ratio", "f8"),
        ("Assortativity Coefficient", "f8"),
        ("Corr. Closeness x Degree", "f8"),
        ("Topology Class", "U12"),
        ("Generator Tool", "U5")
    ])
    for i, f in enumerate(graph_paths):
        diG = nx.read_gpickle(str(f))
        G = diG.to_undirected()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        closeness_centrality = np.array(list(nx.closeness_centrality(G, distance="distance").values()))
        assortativity = nx.degree_pearson_correlation_coefficient(G, weight="distance")
        degree = np.array(list(dict(nx.degree(G)).values()))
        cov = np.cov(np.array([closeness_centrality, degree]))[0,1]
        avg_degree = 2 * n_edges / n_nodes
        max_degree = degree.max()
        max_degree_norm = max_degree / n_nodes
        R = (degree > avg_degree).sum() / n_nodes
        top_class = "Ladder" if max_degree_norm < 0.4 and avg_degree < 3 else "Star or H&S"
        if top_class == "Star or H&S":
            star_hs.append("Star" if R < 0.25 else "H&S")

        data_stats[i] = (n_nodes, avg_degree, max_degree, max_degree_norm, R, assortativity, cov, top_class, G.graph["from"])
    df_stats = pd.DataFrame(data_stats)

    save_figs(df_stats, [
        ( "Number of Nodes", "Average Degree", path_stats.joinpath("avg_degree_{}.pdf".format(path.name)) ),
        ( "Average Degree", "Assortativity Coefficient", path_stats.joinpath("assortativity_{}.pdf".format(path.name)) ),
        ( "Number of Nodes", "Corr. Closeness x Degree", path_stats.joinpath("coor_{}.pdf".format(path.name)) )
    ])

    save_figs(df_stats, [
        ( "Average Degree", "Normalized Maximum Degree", path_stats.joinpath("norm_avg_{}.pdf".format(path.name)) ),
    ])

    non_ladder = df_stats.loc[df_stats["Topology Class"] == "Star or H&S"]
    non_ladder.loc[:, "Topology Class"] = np.array(star_hs, dtype="U12")
    if not non_ladder.empty:
        save_figs(non_ladder, [
            ( "High-Degree Ratio", "Maximum Degree", path_stats.joinpath("r_{}.pdf".format(path.name)) )
        ])
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
    p.add_argument("--type", type=str, default="brite", choices=["zoo", "brite"], help="Get graphs from brite or topology zoo")
    p.add_argument("--raw-zoo-dir", type=str, default="zoo/", help="Path to gml topology zoo file")
    p.add_argument("--path", type=str, default="dataset/", help="Path to save data")
    p.add_argument("--n-interval", type=interval, default=(8,20), help="Interval for number of nodes")
    p.add_argument("--m-interval", type=interval, default=(2,2), help="Interval for connectivity degree")
    p.add_argument("--placement-interval", type=interval, default=(1,1), help="Interval for node placement parameter")
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
    path_stats = os.path.join(args.path, "stats")

    if args.type == "zoo":
        try:
            assert os.path.isdir(args.raw_zoo_dir)
        except AssertionError:
            print("Path {} is not a valid diretory.".format(args.raw_zoo_dir))
            exit(1)
        print("For zoo, parameter 'size' will be ignored")
        pt.create_static_zoo_dataset(os.path.abspath(path), os.path.abspath(args.raw_zoo_dir), args.n_interval, offset=offset)
    else:
        pt.create_static_brite_dataset(os.path.abspath(path), args.size, args.n_interval,
                                       args.m_interval, args.placement_interval, offset=offset)
    get_stats(path, path_stats)
