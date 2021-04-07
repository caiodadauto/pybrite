import os
from pathlib import Path

import numpy as np
import networkx as nx

from tqdm import tqdm
from .parserzoo import get_zoo_graph
from .parserbrite import get_brite_graph
from .utils import add_shortest_path, graph_to_input_target


def batch_files_generator(
    graphdir,
    file_ext,
    n_batch,
    dataset_size=None,
    shuffle=False,
    edge_scaler=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    global_field=None,
):
    graphdir = Path(graphdir)
    if dataset_size is None:
        dataset_size = len(
            [
                f
                for f in os.listdir(str(graphdir))
                if os.path.splitext(f)[1] == "." + file_ext
            ]
        )
    suffix = np.arange(0, dataset_size, 1)
    if shuffle:
        np.random.shuffle(suffix)
    if n_batch > 0:
        slices = np.arange(0, dataset_size, n_batch)
        slices[-1] = dataset_size
    else:
        slices = np.array([0, dataset_size])
    for i in range(1, len(slices)):
        batch_suffix = suffix[slices[i - 1] : slices[i]]
        input_batch, target_batch, raw_input_edge_features, pos_batch = read_from_files(
            graphdir,
            file_ext,
            batch_suffix,
            edge_scaler,
            bidim_solution,
            input_fields,
            target_fields,
            global_field,
        )
        yield input_batch, target_batch, raw_input_edge_features, pos_batch


def read_from_files(
    graphdir,
    file_ext,
    batch_suffix,
    edge_scaler=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    global_field=None,
):
    input_batch = []
    target_batch = []
    pos_batch = []
    for s in batch_suffix:
        digraph = nx.read_gpickle(str(graphdir.joinpath(str(s) + "." + file_ext)))
        input_graph, target_graph, raw_input_edge_features = graph_to_input_target(
            digraph,
            edge_scaler=edge_scaler,
            input_fields=input_fields,
            target_fields=target_fields,
            global_field=global_field,
            bidim_solution=bidim_solution,
        )
        pos = digraph.nodes(data="pos")
        input_batch.append(input_graph)
        target_batch.append(target_graph)
        pos_batch.append(pos)
    return input_batch, target_batch, raw_input_edge_features, pos_batch


def create_brite_graph(
    min_n, min_m, min_placement, max_n, max_m, max_placement, random_state
):
    n = random_state.choice(range(min_n, max_n + 1))
    m = random_state.choice(range(min_m, max_m + 1))
    node_placement = random_state.choice(range(min_placement, max_placement + 1))
    graph = get_brite_graph(n, m, node_placement, random_state=random_state)
    digraph = add_shortest_path(graph, random_state=random_state)
    return digraph


def create_static_zoo_dataset(
    graphdir, gmldir, interval_node, random_state=None, offset=0
):
    # graphs = []
    name = 0
    graphdir = Path(graphdir)
    gmldir = Path(gmldir)
    min_n, max_n = interval_node
    range_nodes = list(range(min_n, max_n + 1))
    random_state = random_state if random_state else np.random.RandomState()
    gml_list = list(gmldir.glob("*.gml"))
    for i in tqdm(range(len(gml_list))):
        G = get_zoo_graph(gml_list[i], range_nodes, random_state=random_state)
        if G:
            digraph = add_shortest_path(G, random_state=random_state)
            nx.write_gpickle(
                digraph, graphdir.joinpath("{:d}.gpickle".format(name + offset))
            )
            name += 1

            # graphs.append((G, gml_list[i].stem))

    with open(graphdir.joinpath("info.dat"), "w") as f:
        f.write("min,max\n")
        f.write("n,{},{}\n".format(min_n, max_n))

    # def draw_graphs(graphs):
    #     import matplotlib.pyplot as plt
    #     i = 0
    #     for G, path in graphs:
    #         i += 1
    #         plt.figure(dpi=130)
    #         nx.draw(G, node_size=40, pos=G.nodes(data="pos"))
    #         plt.legend(loc="best", title=path + " {:.2f}".format(2*G.number_of_edges() / G.number_of_nodes()), handles=[])
    #         plt.savefig(os.path.join("visual", "%s.png"%i))
    #         plt.close()
    # draw_graphs(graphs)


def get_topology_class(digraph):
    g = digraph.to_undirected()
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    degree = np.array(list(dict(nx.degree(g)).values()))
    avg_degree = 2 * n_edges / n_nodes
    max_degree = degree.max()
    max_degree_norm = max_degree / n_nodes
    r = (degree > avg_degree).sum() / n_nodes
    top_class = "ladder" if max_degree_norm < 0.4 and avg_degree < 3 else "star or hs"
    if top_class == "star or hs":
        top_class = "star" if r < 0.25 else "hs"
    return top_class


def create_static_brite_dataset(
    graphdir,
    n_graphs,
    interval_node,
    interval_m=(2, 2),
    interval_placement=(1, 1),
    random_state=None,
    offset=0,
    balanced=True,
):
    graphdir = Path(graphdir)
    min_n, max_n = interval_node
    min_m, max_m = interval_m
    min_placement, max_placement = interval_placement
    random_state = random_state if random_state else np.random.RandomState()
    classes = ["ladder", "star", "hs"]
    if balanced:
        idx = np.floor(np.linspace(0, n_graphs, len(classes) + 1)).astype(int)
        for i in range(1, idx.size):
            c = classes[i - 1]
            print("Get graphs for {} class".format(c))
            for j in tqdm(range(idx[i - 1], idx[i])):
                while True:
                    digraph = create_brite_graph(
                        min_n,
                        min_m,
                        min_placement,
                        max_n,
                        max_m,
                        max_placement,
                        random_state,
                    )
                    if c == get_topology_class(digraph):
                        nx.write_gpickle(
                            digraph,
                            graphdir.joinpath("{:d}.gpickle".format(j + offset)),
                        )
                        break
    else:
        for i in tqdm(range(n_graphs)):
            digraph = create_brite_graph(
                min_n,
                min_m,
                min_placement,
                max_n,
                max_m,
                max_placement,
                random_state,
            )
            nx.write_gpickle(
                digraph, graphdir.joinpath("{:d}.gpickle".format(i + offset))
            )
    with open(graphdir.joinpath("info.dat"), "w") as f:
        f.write("min,max\n")
        f.write("n,{},{}\n".format(min_n, max_n))
        f.write("m,{},{}".format(min_m, max_m))


def batch_brite_generator(
    n_batch,
    interval_node,
    interval_m=(2, 2),
    interval_placement=(1, 1),
    random_state=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    global_field=None,
):
    min_n, max_n = interval_node
    min_m, max_m = interval_m
    min_placement, max_placement = interval_placement
    random_state = random_state if random_state else np.random.RandomState()
    while True:
        input_batch = []
        target_batch = []
        pos_batch = []
        for _ in range(n_batch):
            digraph = create_brite_graph(
                min_n, min_m, min_placement, max_n, max_m, max_placement, random_state
            )
            input_graph, target_graph, raw_input_edge_features = graph_to_input_target(
                digraph,
                bidim_solution=bidim_solution,
                input_fields=input_fields,
                target_fields=target_fields,
                global_field=global_field,
            )
            input_batch.append(input_graph)
            target_batch.append(target_graph)
            pos_batch.append(dict(digraph.nodes(data="pos")))
        yield input_batch, target_batch, raw_input_edge_features, pos_batch
