import os
from pathlib import Path

import numpy as np
import networkx as nx

from tqdm import tqdm
from .parserzoo import get_zoo_graph
from .parserbrite import get_brite_graph
from .utils import add_shortest_path, graph_to_input_target


def batch_files_generator(graphdir, n_batch, shuffle=False,
                          bidim_solution=True, input_fields=None,
                          target_fields=None, global_field=None):
    graphdir = Path(graphdir)
    graph_files = list(sorted(graphdir.glob("*.gpickle"), key=lambda x: int(x.stem)))
    if shuffle:
        np.random.shuffle(graph_files)
    slices = [slice(start, start + n_batch) for start in range(0, len(graph_files), n_batch)]
    rest = len(graph_files) % n_batch
    if rest != 0:
        if rest / n_batch > .4:
            slices.append(slice(slices[-1].stop, len(graph_files)))
        else:
            slices[-1] = slice(slices[-1].start, len(graph_files))
    for batch_slice in slices:
        batch_files = graph_files[batch_slice]
        input_batch, target_batch, pos_batch = read_from_files(batch_files, bidim_solution, input_fields, target_fields, global_field)
        yield input_batch, target_batch, pos_batch

def read_from_files(files, bidim_solution=True, input_fields=None, target_fields=None, global_field=None):
    input_batch = []
    target_batch = []
    pos_batch = []
    for f in files:
        digraph = nx.read_gpickle(f)
        input_graph, target_graph = graph_to_input_target(
            digraph, bidim_solution=bidim_solution, input_fields=input_fields,
            target_fields=target_fields, global_field=global_field)
        pos = digraph.node(data="pos")
        input_batch.append(input_graph)
        target_batch.append(target_graph)
        pos_batch.append(pos)
    return input_batch, target_batch, pos_batch

def create_brite_graph(n, m, node_placement, random_state):
    graph = get_brite_graph(n, m, node_placement, random_state=random_state)
    digraph = add_shortest_path(graph, random_state=random_state)
    return digraph

def create_static_zoo_dataset(graphdir, gmldir, interval_node, random_state=None, offset=0):
    # graphs = []
    name = 0
    graphdir = Path(graphdir)
    gmldir = Path(gmldir)
    min_n, max_n = interval_node
    range_nodes = list(range(min_n, max_n + 1))
    random_state = random_state if random_state else np.random.RandomState()
    gml_list = list(gmldir.glob('*.gml'))
    for i in tqdm(range(len(gml_list))):
        G = get_zoo_graph(gml_list[i], range_nodes, random_state=random_state)
        if G:
            digraph = add_shortest_path(G, random_state=random_state)
            nx.write_gpickle(digraph, graphdir.joinpath("{:d}.gpickle".format(name + offset)))
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

def create_static_brite_dataset(graphdir, n_graphs, interval_node, interval_m=(2, 2), interval_placement=(1, 1), random_state=None, offset=0):
    graphdir = Path(graphdir)
    min_n, max_n = interval_node
    min_m, max_m = interval_m
    min_placement, max_placement = interval_placement
    random_state = random_state if random_state else np.random.RandomState()
    for i in tqdm(range(n_graphs)):
        n = random_state.choice(range(min_n, max_n + 1))
        m = random_state.choice(range(min_m, max_m + 1))
        node_placement = random_state.choice(range(min_placement, max_placement + 1))
        digraph = create_brite_graph(n, m, node_placement, random_state)
        nx.write_gpickle(digraph, graphdir.joinpath("{:d}.gpickle".format(i + offset)))
    with open(graphdir.joinpath("info.dat"), "w") as f:
        f.write("min,max\n")
        f.write("n,{},{}\n".format(min_n, max_n))
        f.write("m,{},{}".format(min_m, max_m))

def batch_brite_generator(n_batch, interval_node,
                          interval_m=(2, 2), interval_placement=(1, 1),
                          random_state=None, bidim_solution=True,
                          input_fields=None, target_fields=None, global_field=None):
    min_n, max_n = interval_node
    min_m, max_m = interval_m
    min_plcement, max_plcement = interval_placement
    random_state = random_state if random_state else np.random.RandomState()
    while True:
        input_batch = []
        target_batch = []
        pos_batch = []
        for _ in range(n_batch):
            n = random_state.choice(range(min_n, max_n + 1))
            m = random_state.choice(range(min_m, max_m + 1))
            node_placement = random_state.choice(range(min_placement, max_placement + 1))
            digraph = create_brite_graph(n, m, node_placement, random_state)
            input_graph, target_graph = graph_to_input_target(
                digraph, bidim_solution=bidim_solution, input_fields=input_fields,
                target_fields=target_fields, global_field=global_field)
            input_batch.append(input_graph)
            target_batch.append(target_graph)
            pos_batch.append(dict(digraph.node(data="pos")))
        yield input_batch, target_batch, pos_batch
