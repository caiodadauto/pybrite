import re
import os
from pathlib import Path

import numpy as np
import networkx as nx
from scipy import spatial
from scipy.optimize import minimize

from .utils import add_ip_prefix, ensure_connection


header = re.compile(r"graph\s*\[")
error_duplicate = re.compile(r"multigraph")
start_edge = re.compile(r"\.*edge\s*\[")
end_edge = re.compile(r"\.*]")
sr_tr = re.compile(r"\((\d+)--(\d+)\)")


def ignore_multigraph(path):
    try:
        graph = nx.read_gml(path, label=None)
    except nx.NetworkXError as e:
        print(e)
        if error_duplicate.search(str(e)):
            source, target = [g for g in sr_tr.search(str(e)).groups()]
            sr = re.compile(r"source %s" % source)
            tr = re.compile(r"target %s" % target)

            with open(str(path) + ".tmp", "w") as outf:
                with open(path, "r") as inf:
                    lines = inf.readlines()
                removed = False
                start_block = False
                end_block = False
                for line in lines:
                    if start_edge.search(line):
                        start_block = True
                        block = []
                        get_sr = False
                        get_tr = False
                    if start_block and end_edge.search(line):
                        end_block = True

                    if end_block:
                        block.append(line)
                        start_block = False
                        end_block = False
                        if not removed and get_sr and get_tr:
                            removed = True
                            continue
                        for block_line in block:
                            outf.write(block_line)
                    elif start_block:
                        block.append(line)
                        if not get_sr and sr.search(line):
                            get_sr = True
                        if not get_tr and tr.search(line):
                            get_tr = True
                    else:
                        outf.write(line)
            os.rename(str(path) + ".tmp", str(path))
            graph = ignore_multigraph(path)
        else:
            raise e
    return graph


def solve_duplicated_nodes(G):
    pos = np.array(list(dict(G.nodes(data="pos")).values()))
    for v, vp in enumerate(pos):
        for u, up in zip(range(v + 1, len(pos)), pos[v + 1 :]):
            if np.all(vp == up):
                vnn = np.array(list(G.neighbors(v)))
                unn = np.array(list(G.neighbors(u)))
                x_scale = 1.3 + np.random.random(1)[0] * 0.3
                y_scale = 1.3 + np.random.random(1)[0] * 0.3
                x_scale *= np.random.choice([1, -1])
                y_scale *= np.random.choice([1, -1])
                new_up = [vp[0] * x_scale, vp[1] * y_scale]
                G.add_node(u, pos=new_up)
                for nn in unn:
                    nnp = pos[nn]
                    G[u][nn]["distance"] = np.linalg.norm(nnp - new_up)


def add_pos(G, random_state):
    def obj(x, v):
        norms = [np.linalg.norm(v[i] - x) for i in range(len(v))]
        return np.sum(norms)

    radius = 600
    empty_polar = []
    nodes_attr = G.nodes(data=True)
    for n, attr in nodes_attr:
        if "Longitude" in attr:
            lon = attr["Longitude"]
            lat = attr["Latitude"]
            G.add_node(
                n, pos=[radius * (lon * np.pi / 180), radius * (lat * np.pi / 180)]
            )
        else:
            empty_polar.append(n)

    if len(empty_polar) < G.number_of_nodes() - 1:
        nodes_attr = G.nodes(data=True)
        for n in empty_polar:
            nn_pos = []
            to_visit = []
            already_visit = []
            do = True
            reference_node = n
            while do:
                for nn in G.neighbors(reference_node):
                    if "pos" in nodes_attr[nn]:
                        nn_pos.append(nodes_attr[nn]["pos"])
                    if not nn in already_visit:
                        to_visit.insert(0, nn)
                already_visit.append(reference_node)
                if len(nn_pos) > 1:
                    nn_pos = np.stack(nn_pos)
                    constraints = []
                    non_nan_pos = np.array(
                        [
                            i
                            for i in list(dict(G.nodes(data="pos")).values())
                            if i is not None
                        ]
                    )
                    for v in non_nan_pos:
                        constraints.append(
                            dict(type="ineq", fun=lambda x: (x - v * 1.5).sum())
                        )
                    n_pos = minimize(
                        obj,
                        random_state.rand(2),
                        args=(nn_pos,),
                        constraints=constraints,
                    ).x
                    do = False
                else:
                    reference_node = to_visit.pop()
            G.add_node(n, pos=list(n_pos))
    else:
        for n, p in nx.spring_layout(G).items():
            G.add_node(n, pos=list(p * radius))


def add_edge_weights(G):
    pos = []
    idx = []
    radius = 600
    for n, p in G.nodes(data="pos"):
        idx.append(n)
        pos.append(p)
    distances = spatial.distance.squareform(spatial.distance.pdist(pos))
    i_, j_ = np.meshgrid(idx, idx, indexing="ij")
    keys = list(zip(i_.ravel(), j_.ravel()))
    values = list(zip(i_.ravel(), j_.ravel(), distances.ravel()))
    all_costs = dict(zip(keys, values))

    cost_edges = []
    edges = list(G.edges)
    for edge in edges:
        if all_costs[edge][-1] < 10e-2:
            all_costs[edge] = (
                all_costs[edge][0],
                all_costs[edge][1],
                radius * (0.1 + np.random.random(1)[0] * 0.5),
            )
        cost_edges.append(all_costs[edge])
    G.add_weighted_edges_from(cost_edges, weight="distance")
    nx.relabel_nodes(
        G, dict(zip(sorted(G.nodes()), range(G.number_of_nodes()))), copy=False
    )


def verify(G, path):
    for s, r, d in G.edges(data="distance"):
        try:
            assert d > 10e-2
        except AssertionError:
            print(
                "In graph {}, edge ({}, {}) with invalid distance, {:.2f}.".format(
                    path.stem, s, r, d
                )
            )
            exit(1)

    nodes = sorted(G.nodes())
    for idx in range(G.number_of_nodes()):
        try:
            assert idx == nodes[idx]
        except AssertionError:
            print(
                "In graph {}, nodes indices are not a sequential, {}.".format(
                    path.stem, nodes
                )
            )
            exit(1)


def get_zoo_graph(path, range_nodes, random_state=None):
    path = Path(path)
    G = ignore_multigraph(path)
    G = ensure_connection(G)
    if not G.number_of_nodes() in range_nodes:
        return None
    if G.number_of_nodes() > 255:
        return None
    add_pos(G)
    add_edge_weights(G)
    add_ip_prefix(G, random_state)
    solve_duplicated_nodes(G)
    G.graph["from"] = "Zoo"
    verify(G, path)
    return G
