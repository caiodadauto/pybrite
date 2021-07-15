import os
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def draw_ip_clusters(digraph, name="", ext="pdf", use_original_pos=False):
    if use_original_pos:
        pos = list(dict(digraph.nodes(data="pos")).values())
    else:
        pos = nx.spring_layout(digraph)

    node_colors = list(dict(digraph.nodes(data="cluster")).values())
    edge_colors = [c for _, _, c in list(digraph.edges(data="cluster"))]
    nx.draw_networkx_nodes(digraph, pos=pos, node_color=node_colors)
    nx.draw_networkx_edges(
        digraph, pos=pos, connectionstyle="arc3,rad=0.2", edge_color=edge_colors
    )
    if name == "":
        plt.savefig("ip_cluster.{}".format(ext))
    else:
        plt.savefig("ip_cluster_{}".format(name) + ".{}".format(ext))
    plt.close()


def plot(fn, name, ext, limits, **kwargs):
    sns.set_style("ticks")
    fig = plt.figure(dpi=400)
    ax = fig.subplots(1, 1, sharey=False)
    kwargs["ax"] = ax
    fn(**kwargs)
    ax.yaxis.grid(True)
    if "x" in kwargs:
        ax.set_xlabel(kwargs["x"], fontsize=18)
    if "y" in kwargs:
        ax.set_ylabel(kwargs["y"], fontsize=18)
    else:
        ax.set_ylabel("Probability", fontsize=18)
    if limits is not None:
        ax.set_yticks(np.arange(limits[0], limits[1], (limits[1] - limits[0]) / 5))
    plt.savefig(name + ".png")
    fig.clear()
    plt.close()


def get_metrics(dir_name):
    i = 0
    paths = [p for p in os.listdir(dir_name) if p.split(".")[-1] == "gpickle"]
    avg_degrees = np.zeros(len(paths))
    graph_sizes = np.zeros(len(paths))
    covs = np.zeros(len(paths))
    assortativities = np.zeros(len(paths))
    corr_closeness_degrees = np.zeros(len(paths))
    for p in tqdm(paths):
        di_graph = nx.read_gpickle(os.path.join(dir_name, p))
        graph = di_graph.to_undirected()
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        avg_degree = 2 * n_edges / n_nodes
        closeness = np.array(list(dict(nx.closeness_centrality(graph)).values()))
        degree = np.array(list(dict(nx.degree(graph)).values()))
        avg_degrees[i] = avg_degree
        graph_sizes[i] = graph.number_of_nodes()
        covs[i] = degree.std() / degree.mean()
        assortativities[i] = nx.degree_assortativity_coefficient(graph)
        corr_closeness_degrees[i] = np.corrcoef(closeness, degree)[0, 1]
        i += 1
    data = pd.DataFrame(
        np.stack(
            [graph_sizes, avg_degrees, covs, assortativities, corr_closeness_degrees],
            axis=1,
        ),
        columns=[
            "Number of Nodes",
            "Avg Degrees",
            "CoV (std / mean)",
            "Assortativity",
            "Corr. Closeness x Node Degrees",
        ],
    )
    return data


def draw_metrics(dir_brite, dir_zoo, ext="png"):
    data_brite = get_metrics(dir_brite)
    data_zoo = get_metrics(dir_zoo)
    data_brite["from"] = data_brite.shape[0] * ["Brite"]
    data_zoo["from"] = data_zoo.shape[0] * ["Zoo"]
    data = pd.concat([data_brite, data_zoo], axis=0)
    scatter = partial(
        sns.scatterplot,
        data=data,
        style="from",
        size="from",
        hue="from",
        legend="full",
        sizes=[50, 80],
        markers=["o", "^"],
        palette="Set1",
    )
    plot(
        sns.histplot,
        "avg_degree_dist",
        ext,
        None,
        data=data,
        x="Avg Degrees",
        hue="from",
        stat="probability",
        palette="Set1",
    )
    plot(
        scatter,
        "avg_degree_x_size",
        ext,
        (0, 8.5),
        x="Number of Nodes",
        y="Avg Degrees",
    )
    plot(
        scatter,
        "CoV_x_size",
        ext,
        None,  # (-0.1, 1.46),
        x="Number of Nodes",
        y="CoV (std / mean)",
    )
    plot(
        scatter,
        "assortativity_x_avg_degree",
        ext,
        None,  # (-1, 0.25),
        x="Avg Degrees",
        y="Assortativity",
    )
    plot(
        scatter,
        "corr_closeness_node_degree_x_size",
        ext,
        (0, 1.5),
        x="Number of Nodes",
        y="Corr. Closeness x Node Degrees",
    )
