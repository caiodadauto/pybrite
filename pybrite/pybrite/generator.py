import subprocess as sub

import numpy as np

from .parserbrite import config_brite, brite_to_graph
from .utils import add_shortest_path, graph_to_input_target
from .paths import GRAPH_BRITE_PATH, BRITE_CONFIG_PATH, SEED_BRITE_PATH, LAST_SEED_BRITE_PATH


def graph_batch_generator(n_batch, interval_node, interval_m=(2, 2), seed=None):
    random_state = np.random.RandomState(seed=seed)
    while True:
        input_batch = []
        target_batch = []
        min_n, max_n = interval_node
        min_m, max_m = interval_m
        for _ in range(n_batch):
            n = random_state.choice(range(min_n, max_n + 1))
            m = random_state.choice(range(min_m, max_m + 1))
            graph = get_graph(n, m, random_state=random_state)
            di_graph, target_node = add_shortest_path(graph, random_state=random_state)
            input_graph, target_graph = graph_to_input_target(di_graph, target_node)
            input_batch.append(input_graph)
            target_batch.append(target_graph)
        yield input_batch, target_batch

def get_graph(n, m=2, random_state=None):
    config_brite(n, m)
    cmd = ['cppgen', BRITE_CONFIG_PATH,  GRAPH_BRITE_PATH, SEED_BRITE_PATH, LAST_SEED_BRITE_PATH]
    _ = sub.run(cmd, stdout=sub.PIPE)
    return brite_to_graph(random_state=random_state)
