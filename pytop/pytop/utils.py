import itertools

import numpy as np
import networkx as nx
from sklearn.cluster import spectral_clustering


def ensure_connection(graph):
    max_n_node = 0
    components = list(nx.connected_components(graph))
    for i in range(len(components)):
        subgraph = graph.subgraph(components[i])
        size = subgraph.number_of_nodes()
        if max_n_node < size:
            max_n_node = size
            max_idx = i
    if len(components) > 0:
        largest_component = graph.subgraph(components[max_idx]).copy()
    nx.relabel_nodes(
        largest_component,
        dict(
            zip(
                sorted(largest_component.nodes()),
                range(largest_component.number_of_nodes()),
            )
        ),
        copy=False,
    )
    return largest_component


def add_ip(digraph, random_state, ratio_upper_range=(0.1, 0.2)):
    graph = digraph.to_undirected()
    n_nodes = graph.number_of_nodes()
    upper_bound = random_state.choice(ratio_upper_range)
    n_subnets = random_state.choice([1, int(upper_bound * n_nodes)])
    labels = spectral_clustering(nx.adjacency_matrix(graph, weight="distance"), n_clusters=n_subnets)
    _, subnet_n_nodes = np.unique(labels, return_counts=True)
    subnet_n_links = np.zeros(n_subnets, dtype=int)
    for i, (u, v) in enumerate(graph.edges()):
        label_u = labels[u]
        label_v = labels[v]
        subnet_n_links[label_u] += 1
        subnet_n_links[label_v] += 1
    ips, prefix_sizes = get_ips(subnet_n_links, random_state)

    subnet_start_idx = np.cumsum(subnet_n_links)
    subnet_start_idx[-1] = 0
    subnet_start_idx = np.roll(subnet_start_idx, 1)
    for p, s in digraph.edges():
        label_p = labels[p]
        digraph.add_edge(
            p,
            s,
            ip=ips[subnet_start_idx[label_p]],
            prefix_size=prefix_sizes[label_p],
            cluster=label_p,
        )
        subnet_start_idx[label_p] += 1
    for n in digraph.nodes():
        label = labels[n]
        digraph.add_node(n, cluster=label)


def get_ips(subnet_sizes, random_state, prefix_range=(20, 28)):
    prefix_sizes = np.zeros(len(subnet_sizes))
    prefixes = np.zeros((len(subnet_sizes), 32))
    for i, subnet_size in enumerate(subnet_sizes):
        prefix = prefixes[i].copy()
        while np.any(np.all(prefix == prefixes, axis=-1)):
            prefix_size = 32
            while 2 ** (32 - prefix_size) - 1 < subnet_size:
                prefix_size = random_state.choice(prefix_range)
            prefix[0:prefix_size] = random_state.choice([0, 1], size=prefix_size)
        prefixes[i] = prefix
        prefix_sizes[i] = prefix_size

    c = 0
    ips = np.zeros((subnet_sizes.sum(), 32))
    for prefix, prefix_size, subnet_size in zip(prefixes, prefix_sizes, subnet_sizes):
        suffix_size = int(32 - prefix_size)
        ips[c] = prefix.copy()
        for i in range(subnet_size):
            ip = prefix.copy()
            while np.any(
                np.all(
                    ip[-suffix_size:] == ips[c : c + subnet_size, -suffix_size:],
                    axis=-1,
                )
            ):
                ip[-suffix_size:] = random_state.choice([0, 1], size=suffix_size)
            ips[c + i, :] = ip.copy()
        c += subnet_size
    return ips, prefix_sizes


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
            solution_edges.extend(list(pairwise(path[end])))
        min_distance.append(
            (node, dict(min_distance_to_end=distance[end], hops_to_end=len(path[end])))
        )
    digraph.add_nodes_from(min_distance)
    digraph.add_edges_from(set_diff(digraph.edges(), solution_edges), solution=False)
    digraph.add_edges_from(solution_edges, solution=True)
    digraph.graph["target"] = end
    return digraph


def graph_to_input_target(
    graph,
    dtype=np.float32,
    scaler=None,
    input_fields=None,
    target_fields=None,
    global_field=None,
    bidim_solution=True,
    random_state=None,
):
    def create_feature(attr, fields, dtype):
        if fields == ():
            return None
        features = []
        for field in fields:
            fattr = np.array(attr[field], dtype=dtype)
            features.append(fattr)
        return np.hstack(features)

    input_node_fields = (
        input_fields["node"] if input_fields and "node" in input_fields else ("pos",)
    )
    input_edge_fields = (
        input_fields["edge"]
        if input_fields and "edge" in input_fields
        else ("ip", "distance")
    )
    target_node_fields = (
        target_fields["node"]
        if target_fields and "node" in target_fields
        else ("min_distance_to_end", "hops_to_end")
    )
    target_edge_fields = (
        target_fields["edge"]
        if target_fields and "edge" in target_fields
        else ("solution",)
    )
    random_state = np.random.RandomState() if random_state is None else random_state

    _graph = graph.copy()
    if scaler is not None:
        d_distance = nx.get_edge_attributes(_graph, "distance")
        d_pos = nx.get_node_attributes(_graph, "pos")
        all_distance = list(d_distance.values())
        all_pos = list(d_pos.values())
        nx.set_edge_attributes(
            _graph, dict(zip(d_distance, scaler(all_distance))), "distance"
        )
        nx.set_node_attributes(_graph, dict(zip(d_pos, scaler(all_pos))), "pos")
        raw_features = (d_distance, d_pos)
    else:
        raw_features = None
    input_graph = _graph.copy()
    target_graph = _graph.copy()

    destination = _graph.graph["target"]
    destination_out_degree = _graph.out_degree(destination)
    destination_interface_idx = random_state.choice(range(destination_out_degree))
    destination_interface = list(_graph.out_edges(destination, data="ip"))[
        destination_interface_idx
    ][-1].astype(dtype)

    for node_index, node_feature in _graph.nodes(data=True):
        input_graph.add_node(
            node_index, features=create_feature(node_feature, input_node_fields, dtype)
        )
        target_graph.add_node(
            node_index, features=create_feature(node_feature, target_node_fields, dtype)
        )

    for sender, receiver, edge_feature in _graph.edges(data=True):
        input_graph.add_edge(
            sender,
            receiver,
            features=create_feature(edge_feature, input_edge_fields, dtype),
        )
        if bidim_solution:
            target_edge = to_one_hot(
                create_feature(edge_feature, target_edge_fields, dtype).astype(
                    np.int32
                ),
                2,
            )[0]
        else:
            target_edge = create_feature(edge_feature, target_edge_fields, dtype)
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = destination_interface
    target_graph.graph["features"] = destination_interface
    return input_graph, target_graph, raw_features
