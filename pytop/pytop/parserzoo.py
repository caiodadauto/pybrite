import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import spatial

from .utils import ip_generator, ensure_connection


header = re.compile(r'graph\s*\[')
error_duplicate = re.compile(r'multigraph')
start_edge = re.compile(r'\.*edge\s*\[')
end_edge = re.compile(r'\.*]')
sr_tr = re.compile(r'\((\d+)--(\d+)\)')

def ignore_multigraph(path):
    try:
        graph = nx.read_gml(path, label=None)
    except nx.NetworkXError as e:
        if error_duplicate.search(str(e)):
            source, target = [g for g in sr_tr.search(str(e)).groups()]
            sr = re.compile(r'source %s'%source)
            tr = re.compile(r'target %s'%target)

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

def add_attr(G, random_state):
    size = 200
    empty_polar = []
    nodes_attr = G.nodes(data=True)
    ip_gen = ip_generator(G.number_of_nodes(), random_state=random_state)
    for n, attr in nodes_attr:
        if 'Longitude' in  attr:
            lon = attr['Longitude']
            lat = attr['Latitude']
            G.add_node(n, pos=[size * (lon * np.pi / 180), size * (lat * np.pi / 180)], ip=next(ip_gen, None))
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
            while(do):
                for nn in G.neighbors(reference_node):
                    if "pos" in nodes_attr[nn]:
                        nn_pos.append(nodes_attr[nn]["pos"])
                    if not nn in already_visit:
                        to_visit.insert(0, nn)
                already_visit.append(reference_node)
                if len(nn_pos) > 1:
                    nn_pos = np.stack(nn_pos)
                    A = (nn_pos[0] - nn_pos[1:]) * 2
                    n_pos = np.dot( np.linalg.pinv(A), (nn_pos[0]**2 - nn_pos[1:]**2).sum(axis=1) )
                    do = False
                else:
                    reference_node = to_visit.pop()
            G.add_node(n, pos=list(n_pos), ip=next(ip_gen, None))
    else:
        for n, p in nx.spring_layout(G).items():
            G.add_node(n, pos=list(p * size), ip=next(ip_gen, None))

    pos = []
    idx = []
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
        cost_edges.append(all_costs[edge])
    G.add_weighted_edges_from(cost_edges, weight="distance")


def get_zoo_graph(path, random_state=None):
    path = Path(path)
    G = ignore_multigraph(path)
    G = ensure_connection(G)
    add_attr(G, random_state)
    return G
