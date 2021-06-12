import itertools

import numpy as np
import networkx as nx


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
    return largest_component


def add_ip(digraph, random_state):
    graph = digraph.to_undirected()
    n_links = graph.number_of_edges()
    ips = get_ips(n_links, random_state)
    for i, (u, v) in enumerate(graph.edges()):
        digraph.add_edge(u, v, ip=ips[i, 0])
        digraph.add_edge(v, u, ip=ips[i, 1])


def get_ips(n_links, random_state, prefix_range=(20, 28)):
    subnet_sizes = []
    _n_links = n_links
    while _n_links > 0:
        subnet_size = random_state.randint(1, _n_links + 1)
        subnet_sizes.append(subnet_size)
        _n_links -= subnet_size

    prefixes = np.zeros((len(subnet_sizes), 33))
    for i, subnet_size in enumerate(subnet_sizes):
        prefix = prefixes[i].copy()
        while np.any(np.all(prefix == prefixes, axis=-1)):
            prefix_size = 32
            while 2 ** (32 - prefix_size) - 1 < subnet_size * 2:
                prefix_size = random_state.choice(prefix_range)
            prefix[0:prefix_size] = random_state.choice([0, 1], size=prefix_size)
        prefix[-1] = prefix_size
        prefixes[i] = prefix

    c = 0
    ips = np.zeros((n_links, 2, 32))
    for prefix, subnet_size in zip(prefixes, subnet_sizes):
        prefix_size = prefix[-1]
        prefix = prefix[0:-1]
        suffix_size = int(32 - prefix_size)
        ips[c:c + subnet_size, :] = prefix.copy()
        for i in range(subnet_size):
            for j in [0, 1]:
                ip = prefix.copy()
                while np.any(
                    np.all(
                        ip[-suffix_size:] == ips[c:c + subnet_size, :, -suffix_size:],
                        axis=-1,
                    )
                ):
                    ip[-suffix_size:] = random_state.choice([0, 1], size=suffix_size)
                ips[c + i, j, :] = ip.copy()
        c += subnet_size
    return ips


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
