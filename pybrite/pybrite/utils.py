import itertools

import numpy as np
import networkx as nx

def ip_generator(size, random_state=None):
    zero_ip = np.zeros(32)
    ip_set = np.array([zero_ip] * size)
    random_state = random_state if random_state else np.random.RandomState()
    for i in range(size):
        ip = zero_ip
        while( np.any([np.all(ip_bool) for ip_bool in ip_set == ip]) ):
            ip = random_state.choice([0,1], size=32)
        ip_set[i] = ip
        yield np.array(ip, dtype=float)

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def set_diff(seq0, seq1):
    return list(set(seq0) - set(seq1))

def to_one_hot(indices, max_value):
    one_hot = np.eye(max_value)[indices]
    return one_hot

def add_shortest_path(graph, random_state):
    random_state = random_state if random_state else np.random.RandomState()

    # Map from node pairs to the length of their shortest path.
    all_paths = nx.all_pairs_dijkstra(graph, weight="distance")
    end = random_state.choice(graph.nodes())

    # Creates a directed graph, to store the directed path from start to end.
    digraph = graph.to_directed()

    # Add "solution" attributes to the edges.
    solution_edges = []
    min_distance = []
    for node, (distance, path) in all_paths:
        if node != end:
            solution_edges.extend(list( pairwise(path[end]) ))
        min_distance.append((node, dict(min_distance_to_end=distance[end], hops_to_end=len(path[end]))))
    digraph.add_nodes_from(min_distance)
    digraph.add_edges_from(set_diff(digraph.edges(), solution_edges), solution=False)
    digraph.add_edges_from(solution_edges, solution=True)
    digraph.graph["target"] = end
    return digraph

def graph_to_input_target(graph, bidim_solution=True, input_fields=None, target_fields=None, global_field=None):
    def create_feature(attr, fields):
        if fields == ():
            return None
        return np.hstack( [np.array(attr[field], dtype=float) for field in fields] )

    input_node_fields = input_fields["node"] if input_fields and "node" in input_fields else ("ip",)
    input_edge_fields = input_fields["edge"] if input_fields and "edge" in input_fields else ("distance",)
    target_node_fields = target_fields["node"] if target_fields and "node" in target_fields else ("min_distance_to_end", "hops_to_end")
    target_edge_fields = target_fields["edge"] if target_fields and "edge" in target_fields else ("solution",)

    input_graph = graph.copy()
    target_graph = graph.copy()

    end = graph.graph["target"]
    for node_index, node_feature in graph.nodes(data=True):
        if node_index == end:
            end_node = node_feature[global_field if global_field else "ip"]
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields))
        target_graph.add_node(
            node_index, features=create_feature(node_feature, target_node_fields))

    for sender, receiver, features in graph.edges(data=True):
        input_graph.add_edge(
            sender, receiver, features=create_feature(features, input_edge_fields))

        if bidim_solution:
            target_edge = to_one_hot(
                create_feature(features, target_edge_fields).astype(int), 2)[0]
        else:
            target_edge = create_feature(features, target_edge_fields)
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = end_node
    target_graph.graph["features"] = end_node
    return input_graph, target_graph
