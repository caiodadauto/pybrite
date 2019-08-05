import re
import os

# import graph_tool as gt
import networkx as nx

from .paths import GRAPH_BRITE_PATH, BRITE_CONFIG_PATH


def config_brite(N, m=2):
    node_line = re.compile(r'^(\s*N\s*=\s*\d+)')
    neighbor_line = re.compile(r'^(\s*m\s*=\s*\d+)')
    with open(BRITE_CONFIG_PATH + ".tmp", "w") as out_f:
        with open(BRITE_CONFIG_PATH, 'r') as in_f:
            lines = in_f.readlines()
            for line in lines:
                if node_line.search(line):
                    out_f.write("\tN = %s\n"%N)
                elif neighbor_line.search(line):
                    out_f.write("\tm = %s\n"%m)
                else:
                    out_f.write(line)
        os.rename(BRITE_CONFIG_PATH + ".tmp", BRITE_CONFIG_PATH)

def brite_to_graph():
    # G = gt.Graph(directed=False)
    # G.vp.pos = G.new_vertex_property("vector<float>")
    # G.ep.weight = G.new_edge_property("float")
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

                # G.add_vertex(n_nodes)
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
                G.add_node(v, pos=[x_pos, y_pos])

                # v = G.vertex(int(features[0]) - node_offset)
                # G.vp.pos[v] = [x_pos, y_pos]
            elif is_edge:
                if not line.strip():
                    is_edge = False
                    continue
                features = re.split(r'\s', line)

                sender = int(features[1]) - node_offset
                receiver = int(features[2]) - node_offset
                weight = float(features[3])
                G.add_edge(sender, receiver, distance=weight)

                # sender = G.vertex(int(features[1]) - node_offset)
                # receiver = G.vertex(int(features[2]) - node_offset)
                # e = G.edge(sender, receiver)
                # G.ep.weight[e] = weight
    return G
