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


def add_ip_prefix(G, random_state):
    # idx = {}
    edges = G.edges()
    # degrees = G.degree()
    # node_interfaces = {}
    prefixes = get_prefix(G.number_of_edges(), random_state=random_state)
    # for n, d in degrees:
    #     idx[n] = 0
    #     node_interfaces[n] = np.zeros((d, 32))
    for i, (p, s) in enumerate(edges):
        G.add_edge(p, s, prefix=prefixes[i])
    #     node_interfaces[p][idx[p]] = prefixes[i]
    #     node_interfaces[s][idx[s]] = prefixes[i]
    #     idx[p] += 1
    #     idx[s] += 1
    # node_params = list(
    #     map(lambda t: (t[0], dict(interfaces=t[1])), tuple(node_interfaces.items()))
    # )
    # G.add_nodes_from(node_params)


def get_prefix(n_subnets, range_subnet_mask=(24, 28), random_state=None):
    random_state = random_state if random_state else np.random.RandomState()
    prefixes = np.zeros((n_subnets, 32))
    for i in range(n_subnets):
        prefix = prefixes[i].copy()
        while np.any([np.all(cmp) for cmp in prefix == prefixes]):
            subnet_mask = random_state.choice(range_subnet_mask)
            prefix[0:subnet_mask] = random_state.choice([0, 1], size=subnet_mask)
        prefixes[i] = prefix
    return prefixes


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
):
    def create_feature(attr, fields, dtype):
        if fields == ():
            return None
        features = []
        for field in fields:
            fattr = np.array(attr[field], dtype=dtype)
            # if field == "interfaces":
            #     fattr = fattr.T
            features.append(fattr)
        return np.hstack(features)

    input_node_fields = (
        input_fields["node"] if input_fields and "node" in input_fields else ("pos",)
    )
    input_edge_fields = (
        input_fields["edge"]
        if input_fields and "edge" in input_fields
        else ("prefix", "distance")
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

    end = _graph.graph["target"]
    for node_index, node_feature in _graph.nodes(data=True):
        if node_index == end:
            end_node = np.array(
                list(_graph.in_edges(node_index, data="prefix"))[0][-1], dtype=dtype
            )
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

    input_graph.graph["features"] = end_node
    target_graph.graph["features"] = end_node
    return input_graph, target_graph, raw_features
