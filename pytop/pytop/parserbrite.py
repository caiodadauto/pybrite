import re
import os
import subprocess as sub

import networkx as nx

from .utils import ip_generator
from .paths import GRAPH_BRITE_PATH, BRITE_CONFIG_PATH, SEED_BRITE_PATH, LAST_SEED_BRITE_PATH


def config_brite(N, m=2, node_placement=1):
    node_line = re.compile(r'^(\s*N\s*=\s*\d+)')
    neighbor_line = re.compile(r'^(\s*m\s*=\s*\d+)')
    placement_line = re.compile(r'^(\s*NodePlacement\s*=\s*\d+)')
    with open(BRITE_CONFIG_PATH + ".tmp", "w") as out_f:
        with open(BRITE_CONFIG_PATH, 'r') as in_f:
            lines = in_f.readlines()
            for line in lines:
                if node_line.search(line):
                    out_f.write("\tN = %s\n"%N)
                elif neighbor_line.search(line):
                    out_f.write("\tm = %s\n"%m)
                elif placement_line.search(line):
                    out_f.write("\tNodePlacement = %s\n"%node_placement)
                else:
                    out_f.write(line)
    os.rename(BRITE_CONFIG_PATH + ".tmp", BRITE_CONFIG_PATH)

def brite_to_graph(random_state=None):
    G = nx.Graph();

    is_node = False
    is_edge = False
    node_offset = -1
    topology_line = re.compile(r'^(Topology)', re.I)
    graph_size = re.compile(r'\(\s*(\d+)\s*Nodes,\s*(\d+)\s*Edges\s*\)', re.I)
    start_nodes = re.compile(r'^(\s*Nodes\s*:*\s*)', re.I)
    start_edges = re.compile(r'^(\s*Edges\s*:*\s*)', re.I)
    with open(GRAPH_BRITE_PATH + ".brite", "r") as brite_file:
        while True:
            line = brite_file.readline()
            if not line:
                break
            if topology_line.search(line):
                n_nodes, n_edges = int(graph_size.search(line).groups()[0]), int(graph_size.search(line).groups()[1])
                ip_gen = ip_generator(n_nodes, random_state=random_state)
            elif start_nodes.search(line):
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

                features = re.split(r'\s', line)
                if node_offset < 0:
                    node_offset = int(features[0])
                v = int(features[0]) - node_offset
                x_pos = float(features[1])
                y_pos = float(features[2])
                G.add_node(v, pos=[x_pos, y_pos], ip=next(ip_gen, None))
            elif is_edge:
                if not line.strip():
                    is_edge = False
                    continue
                features = re.split(r'\s', line)

                sender = int(features[1]) - node_offset
                receiver = int(features[2]) - node_offset
                weight = float(features[3])
                G.add_edge(sender, receiver, distance=weight)
    G.graph["from"] = "Brite"
    return G

def get_brite_graph(n, m, node_placement, random_state):
    config_brite(n, m, node_placement)
    cmd = ['cppgen', BRITE_CONFIG_PATH,  GRAPH_BRITE_PATH, SEED_BRITE_PATH, LAST_SEED_BRITE_PATH]
    _ = sub.run(cmd, stdout=sub.PIPE)
    return brite_to_graph(random_state=random_state)
