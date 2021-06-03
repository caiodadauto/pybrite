import os
import sys
from pathlib import Path

import numpy as np
import networkx as nx
if sys.version_info >= (3, 8):
    from networkx import read_gpickle
else:
    from pickle5 import load

    def read_gpickle(path):
        with open(path, 'rb') as f:
            g = load(f)
        return g

from tqdm import tqdm
from .parserzoo import get_zoo_graph
from .parserbrite import create_brite_graph
from .utils import add_shortest_path, graph_to_input_target


def batch_files_generator(
    graphdir,
    file_ext,
    n_batch,
    trunc_ip=False,
    one_hot=False,
    dataset_size=None,
    shuffle=False,
    scaler=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    global_field=None,
    random_state=None,
    seen_graphs=0,
    dtype=np.float32,
):
    random_state = np.random.RandomState() if random_state is None else random_state
    graphdir = Path(graphdir)
    if dataset_size is None:
        suffix = [
            os.path.splitext(f)[0] for f in os.listdir(
                str(graphdir)) if os.path.splitext(f)[1] == "." + file_ext
        ]
    else:
        suffix = np.arange(0, dataset_size, 1)
    if shuffle:
        random_state.shuffle(suffix)
    if seen_graphs > 0:
        suffix = suffix[seen_graphs + 1:]
        dataset_size = len(suffix)
    if n_batch > 0:
        slices = np.arange(0, dataset_size, n_batch)
        slices[-1] = dataset_size
    else:
        slices = np.array([0, dataset_size])
    for i in range(1, len(slices)):
        batch_suffix = suffix[slices[i - 1]: slices[i]]
        input_batch, target_batch, raw_input_edge_features, pos_batch = read_from_files(
            graphdir,
            file_ext,
            batch_suffix,
            scaler,
            bidim_solution,
            input_fields,
            target_fields,
            global_field,
            one_hot=one_hot,
            trunc_ip=trunc_ip,
            dtype=dtype,
        )
        yield input_batch, target_batch, raw_input_edge_features, pos_batch


def read_from_files(
    graphdir,
    file_ext,
    batch_suffix,
    scaler=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    global_field=None,
    trunc_ip=False,
    one_hot=False,
    dtype=np.float32,
):
    input_batch = []
    target_batch = []
    pos_batch = []
    for s in batch_suffix:
        digraph = read_gpickle(str(graphdir.joinpath(str(s) + "." + file_ext)))
        input_graph, target_graph, raw_input_edge_features = graph_to_input_target(
            digraph,
            trunc_ip=trunc_ip,
            one_hot=one_hot,
            scaler=scaler,
            input_fields=input_fields,
            target_fields=target_fields,
            global_field=global_field,
            bidim_solution=bidim_solution,
            dtype=dtype,
        )
        pos = digraph.nodes(data="pos")
        input_batch.append(input_graph)
        target_batch.append(target_graph)
        pos_batch.append(pos)
    return input_batch, target_batch, raw_input_edge_features, pos_batch


def create_static_zoo_dataset(
    graphdir, gmldir, interval_node, random_state=None, offset=0
):
    # graphs = []
    name = 0
    gmldir = Path(gmldir)
    graphdir = Path(graphdir)
    min_n, max_n = interval_node
    range_nodes = list(range(min_n, max_n + 1))
    random_state = random_state if random_state else np.random.RandomState()
    gml_list = list(gmldir.glob("*.gml"))
    for i in tqdm(range(len(gml_list))):
        G = get_zoo_graph(gml_list[i], range_nodes, random_state=random_state)
        if G:
            digraph = add_shortest_path(G, random_state=random_state)
            nx.write_gpickle(
                digraph, graphdir.joinpath(
                    "{:d}.gpickle".format(name + offset))
            )
            name += 1
            # graphs.append((G, gml_list[i].stem))
    # print(name, len(gml_list))
    # print(name / len(gml_list))
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
    interval_composition,
    class_ratio,
    main_plane_size=1000,
    random_state=None,
    offset=0,
    balanced=True,
):
    graphdir = Path(graphdir)
    classes = ["hs", "ladder", "star"]
    random_state = random_state if random_state else np.random.RandomState()
    main_bar = tqdm(total=n_graphs, desc="Generting graphs")
    if balanced:
        bars = {}
        n_hs = int(np.ceil(n_graphs * class_ratio[0]))
        n_l = int(np.ceil(n_graphs * class_ratio[1]))
        n_s = n_graphs - (n_hs + n_l)
        n_graphs_per_class = dict(zip(classes, [n_hs, n_l, n_s]))
        for i in range(len(classes)):
            bars[classes[i]] = tqdm(
                total=n_graphs_per_class[classes[i]],
                desc="Getting {}".format(classes[i]),
            )

        total_top = 0
        iter_classes = iter(classes)
        top_class = next(iter_classes)
        got_top = dict(zip(classes, len(classes) * [0]))
        main_bar.set_postfix(c=top_class)
        while total_top < n_graphs:
            valid_graph = False
            while not valid_graph:
                digraph = create_brite_graph(
                    interval_node,
                    interval_composition,
                    main_plane_size,
                    random_state,
                    top_class,
                )
                c = get_topology_class(digraph)
                if got_top[c] < n_graphs_per_class[c]:
                    nx.write_gpickle(
                        digraph,
                        graphdir.joinpath(
                            "{:d}.gpickle".format(total_top + offset)),
                    )
                    total_top += 1
                    got_top[c] += 1
                    bars[c].update()
                    main_bar.update()
                    valid_graph = True
                elif c == top_class:
                    top_class = next(iter_classes)
                    main_bar.set_postfix(c=top_class)
    else:
        for i in tqdm(range(n_graphs)):
            digraph = create_brite_graph(
                interval_node,
                interval_composition,
                main_plane_size,
                random_state,
            )
            nx.write_gpickle(
                digraph, graphdir.joinpath("{:d}.gpickle".format(i + offset))
            )
    with open(graphdir.joinpath("info.dat"), "w") as f:
        f.write("min,max\n")
        f.write("n,{},{}\n".format(interval_node[0], interval_node[1]))
        f.write(
            "composition,{},{}".format(
                interval_composition[0], interval_composition[1])
        )


def batch_brite_generator(
    n_batch,
    interval_node,
    interval_composition,
    one_hot=False,
    trunc_ip=False,
    main_plane_size=1000,
    random_state=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    global_field=None,
    top_class=None,
    dtype=np.float32,
):
    random_state = random_state if random_state else np.random.RandomState()
    while True:
        input_batch = []
        target_batch = []
        pos_batch = []
        for _ in range(n_batch):
            digraph = create_brite_graph(
                interval_node,
                interval_composition,
                main_plane_size,
                random_state,
                top_class,
            )
            input_graph, target_graph, raw_input_edge_features = graph_to_input_target(
                digraph,
                trunc_ip=trunc_ip,
                one_hot=one_hot,
                bidim_solution=bidim_solution,
                input_fields=input_fields,
                target_fields=target_fields,
                global_field=global_field,
                dtype=dtype,
            )
            input_batch.append(input_graph)
            target_batch.append(target_graph)
            pos_batch.append(dict(digraph.nodes(data="pos")))
        yield input_batch, target_batch, raw_input_edge_features, pos_batch
