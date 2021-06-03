import re
import os
import subprocess as sub

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from .utils import add_ip_prefix, add_shortest_path
from .paths import (
    GRAPH_BRITE_PATH,
    BRITE_CONFIG_PATH,
    SEED_BRITE_PATH,
    LAST_SEED_BRITE_PATH,
)


def config_brite(n, m, node_placement, main_plane_size, inner_plane_size):
    node_line = re.compile(r"^(\s*N\s*=\s*\d+)")
    neighbor_line = re.compile(r"^(\s*m\s*=\s*\d+)")
    placement_line = re.compile(r"^(\s*NodePlacement\s*=\s*\d+)")
    main_plane_line = re.compile(r"^(\s*HS\s*=\s*\d+)")
    inner_plane_line = re.compile(r"^(\s*LS\s*=\s*\d+)")
    with open(BRITE_CONFIG_PATH + ".tmp", "w") as out_f:
        with open(BRITE_CONFIG_PATH, "r") as in_f:
            lines = in_f.readlines()
            for line in lines:
                if node_line.search(line):
                    out_f.write("\tN = %s\n" % n)
                elif neighbor_line.search(line):
                    out_f.write("\tm = %s\n" % m)
                elif placement_line.search(line):
                    out_f.write("\tNodePlacement = %s\n" % node_placement)
                elif main_plane_line.search(line):
                    out_f.write("\tHS = %s\n" % main_plane_size)
                elif inner_plane_line.search(line):
                    out_f.write("\tLS = %s\n" % inner_plane_size)
                else:
                    out_f.write(line)
    os.rename(BRITE_CONFIG_PATH + ".tmp", BRITE_CONFIG_PATH)


def brite_to_graph(random_state=None):
    G = nx.Graph()

    is_node = False
    is_edge = False
    node_offset = -1
    start_nodes = re.compile(r"^(\s*Nodes\s*:*\s*)", re.I)
    start_edges = re.compile(r"^(\s*Edges\s*:*\s*)", re.I)
    with open(GRAPH_BRITE_PATH + ".brite", "r") as brite_file:
        while True:
            line = brite_file.readline()
            if not line:
                break
            if start_nodes.search(line):
                is_node = True
                continue
            elif start_edges.search(line):
                is_edge = True
                if is_node:
                    is_node = False
                continue

            if is_node:
                if not line.strip():
                    is_node = False
                    continue

                features = re.split(r"\s", line)
                if node_offset < 0:
                    node_offset = int(features[0])
                v = int(features[0]) - node_offset
                x_pos = float(features[1])
                y_pos = float(features[2])
                G.add_node(v, pos=[x_pos, y_pos])
            elif is_edge:
                if not line.strip():
                    is_edge = False
                    continue
                features = re.split(r"\s", line)

                sender = int(features[1]) - node_offset
                receiver = int(features[2]) - node_offset
                weight = float(features[3])  # Length
                G.add_edge(sender, receiver, distance=weight)
    G.graph["from"] = "Brite"
    return G


def get_brite_graph(
    n, m, node_placement, main_plane_size, inner_plane_size, random_state
):
    config_brite(n, m, node_placement, main_plane_size, inner_plane_size)
    cmd = [
        "cppgen",
        BRITE_CONFIG_PATH,
        GRAPH_BRITE_PATH,
        SEED_BRITE_PATH,
        LAST_SEED_BRITE_PATH,
    ]
    _ = sub.run(cmd, stdout=sub.PIPE)
    return brite_to_graph(random_state=random_state)


def get_n_nodes_to_connected(size, top_class, random_state):
    if top_class == "star":
        return random_state.choice(range(1, 4), p=[0.6, 0.2, 0.2])
    return int(size * 0.2)


def get_m(top_class, random_state, flag=0, base=None):
    if top_class == "ladder":
        p = [0.65, 0.20, 0.12, 0.02, 0.01]
    elif top_class == "star":
        if flag == 0:
            p = [.95, 0.05, 0.0, 0.0, 0.0]
        elif flag == 1:
            p = [0.05, 0.15, 0.8]
            values = [.8, 0.9, 1.0]
        else:
            p = [0.8, 0.15, 0.05]
            values = [.02, 0.05, 0.1]
    elif top_class == "hs":
        p = [0.25, 0.5, 0.20, 0.025, 0.025]
    else:
        p = [0.2, 0.2, 0.2, 0.2, 0.2]
    if flag == 0 or top_class != "star":
        m = random_state.choice(range(1, 6), p=p)
    else:
        m = random_state.choice(values, p=p)
        m = int(np.ceil(m * base))
    return m


def create_composition(
    top_class,
    min_n,
    max_n,
    sub_plane_size,
    sub_inner_plane_size,
    composition,
    random_state,
    ratio_gap=0.05,
):
    partial_max_n = max_n // composition
    partial_min_n = min_n // composition
    if partial_min_n < 6:
        raise ValueError("min_n / composition < 6")

    side = 1
    stage = 1
    max_gx = 0
    max_gy = 0
    placement = 2
    final_size = 0
    n_composition_to_add = 3
    last_G = None
    G = nx.Graph()
    composition_labels = []
    for _ in range(composition):
        n = random_state.choice(range(partial_min_n, partial_max_n + 1))
        m = get_m(top_class, random_state)
        new_labels = list(range(final_size, n + final_size))
        partial_G = get_brite_graph(
            n, m, placement, sub_plane_size, sub_inner_plane_size, random_state
        )
        pos = np.array(list(dict(partial_G.nodes(data="pos")).values()))
        gx = pos[:, 0].max()
        gy = pos[:, 1].max()
        if last_G is not None:
            transformed_partial_G = partial_G.copy()
            if gx > max_gx:
                max_gx = gx
            if gy > max_gy:
                max_gy = gy
            if stage == n_composition_to_add:
                pos[:, 0] += (max_gx * side) + max_gx * ratio_gap
                pos[:, 1] += (max_gy * side) + max_gy * ratio_gap
                stage = 1
                side += 1
                n_composition_to_add = 2 * side + 1
            elif stage <= (n_composition_to_add - 1) / 2:
                y_shift = stage - 1
                pos[:, 0] += (max_gx * side) + max_gx * ratio_gap
                pos[:, 1] += (max_gy * y_shift) + max_gy * ratio_gap
                stage += 1
            else:
                x_shift = -(stage - 1 - (n_composition_to_add - 1) / 2)
                pos[:, 0] += (max_gx * x_shift) + max_gx * ratio_gap
                pos[:, 1] += (max_gy * side) + max_gy * ratio_gap
                stage += 1
            nx.set_node_attributes(
                transformed_partial_G, dict(zip(range(n), pos)), "pos"
            )
            nx.relabel.relabel_nodes(
                transformed_partial_G, dict(zip(range(n), new_labels)), copy=False
            )
            G = nx.compose(transformed_partial_G, G)
        else:
            G = nx.compose(partial_G, G)
        final_size += n
        last_G = partial_G
        composition_labels.append(new_labels)
    return G, final_size, composition_labels


def add_barabasi_edges(G, top_class, composition_labels, random_state):
    existed_nodes = []
    pos = dict(G.nodes(data="pos"))
    composition = len(composition_labels)
    more_connections = composition - 1
    for i in range(1, composition):
        if i == more_connections:
            flag = 1
        else:
            flag = 2
        existed_nodes += composition_labels[i - 1]
        incoming_labels = composition_labels[i]
        incoming_size = len(incoming_labels)
        k = get_n_nodes_to_connected(incoming_size, top_class, random_state)
        labels_to_be_connected = random_state.choice(incoming_labels, k, replace=False)
        attention_degrees = np.array(
            G.degree(existed_nodes), dtype=[("node", np.int32), ("degree", np.int32)]
        )
        degree_sum = attention_degrees["degree"].sum()
        connection_prob = attention_degrees["degree"] / degree_sum
        for u in labels_to_be_connected:
            m = get_m(top_class, random_state, flag, base=len(connection_prob))
            to_idx = random_state.choice(
                len(attention_degrees), m, p=connection_prob, replace=False
            )
            for v_idx in to_idx:
                v = attention_degrees["node"][v_idx]
                length = np.linalg.norm([pos[u][0] - pos[v][0], pos[u][1] - pos[v][1]])
                G.add_edge(u, v, distance=length)
                degree_sum += 1
                attention_degrees["degree"][v_idx] += 1
                connection_prob = attention_degrees["degree"] / degree_sum


def create_brite_graph(
    interval_node,
    interval_composition,
    main_plane_size,
    random_state,
    top_class=None,
):
    min_n, max_n = interval_node
    min_composition, max_composition = interval_composition
    composition = random_state.choice(range(min_composition, max_composition + 1))
    sub_plane_size = main_plane_size // composition
    sub_inner_plane_size = int(0.1 * sub_plane_size)
    G, num_of_nodes, composition_labels = create_composition(
        top_class,
        min_n,
        max_n,
        sub_plane_size,
        sub_inner_plane_size,
        composition,
        random_state,
    )
    # nx.draw(G, pos=G.nodes(data="pos"), node_size=50)
    # plt.savefig("/home/caio/Documents/university/Ph.D./topology/before_adding_nodes.pdf")
    add_barabasi_edges(G, top_class, composition_labels, random_state)
    add_ip_prefix(G, random_state)
    digraph = add_shortest_path(G, random_state=random_state)
    return digraph
