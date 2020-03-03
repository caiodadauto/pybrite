import subprocess as sub
from pathlib import Path

import numpy as np
import networkx as nx

from .parserbrite import config_brite, brite_to_graph
from .utils import add_shortest_path, graph_to_input_target
from .paths import GRAPH_BRITE_PATH, BRITE_CONFIG_PATH, SEED_BRITE_PATH, LAST_SEED_BRITE_PATH


def create_graph(n, m, random_state):
    graph = get_graph(n, m, random_state=random_state)
    digraph = add_shortest_path(graph, random_state=random_state)
    return digraph

def create_static_dataset(graphdir, n_graphs, interval_node, interval_m=(2, 2), random_state=None):
    graphdir = Path(graphdir)
    min_n, max_n = interval_node
    min_m, max_m = interval_m
    random_state = random_state if random_state else np.random.RandomState()
    for i in range(n_graphs):
        n = random_state.choice(range(min_n, max_n + 1))
        m = random_state.choice(range(min_m, max_m + 1))
        digraph = create_graph(n, m, random_state)
        nx.write_gpickle(digraph, graphdir.joinpath("{:d}.gpickle".format(i)))
    with open(graphdir.joinpath("info.dat"), "w") as f:
        f.write("min,max\n")
        f.write("n,{},{}\n".format(min_n, max_n))
        f.write("m,{},{}".format(min_m, max_m))


def get_graph(n, m=2, random_state=None):
    config_brite(n, m)
    cmd = ['cppgen', BRITE_CONFIG_PATH,  GRAPH_BRITE_PATH, SEED_BRITE_PATH, LAST_SEED_BRITE_PATH]
    _ = sub.run(cmd, stdout=sub.PIPE)
    return brite_to_graph(random_state=random_state)

def read_from_files(files, bidim_solution=True,
                    input_fields=None, target_fields=None, global_field=None):
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

def graph_batch_files_generator(graphdir, n_batch, shuffle=False,
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

def graph_batch_generator(n_batch, interval_node,
                          interval_m=(2, 2), random_state=None,
                          bidim_solution=True, input_fields=None,
                          target_fields=None, global_field=None):
    min_n, max_n = interval_node
    min_m, max_m = interval_m
    random_state = random_state if random_state else np.random.RandomState()
    while True:
        input_batch = []
        target_batch = []
        pos_batch = []
        for _ in range(n_batch):
            n = random_state.choice(range(min_n, max_n + 1))
            m = random_state.choice(range(min_m, max_m + 1))
            digraph = create_graph(n, m, random_state)
            input_graph, target_graph = graph_to_input_target(
                digraph, bidim_solution=bidim_solution, input_fields=input_fields,
                target_fields=target_fields, global_field=global_field)
            input_batch.append(input_graph)
            target_batch.append(target_graph)
            pos_batch.append(dict(digraph.node(data="pos")))
        yield input_batch, target_batch, pos_batch
