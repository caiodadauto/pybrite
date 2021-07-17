#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <limits>
#include "_min_weights.h"


std::string edge_to_str(int a[2])
{
    return std::to_string(a[0]) + std::to_string(a[1]);
}

void dfs_to_target(int source, int target, const std::vector<std::vector<float> >& WU, std::vector<short>& visited, w_table& min_edge_weights, std::vector<int>& path, std::vector<float>& weights)
{
    visited[source] = 1;
    path.push_back(source);
    if(source == target) {

        for(auto n: path) {
            std::cout << n << "  ";
        }
        std::cout << std::endl;
        for(auto n: weights) {
            std::cout << n << "  ";
        }
        std::cout << std::endl;

        int path_size = path.size(); 
        for(int i = 0; i < path_size - 1; i++) {
            int edge[2] = {path[i], path[i + 1]};
            float partial_w = weights[path_size - 1] - weights[i];
            if(min_edge_weights[edge_to_str(edge)] >= partial_w) {
                min_edge_weights[edge_to_str(edge)] = partial_w;
                std::cout << "(" << edge[0] << ", " << edge[1] << ")" << " --> " << partial_w << std::endl;
            }
        }
        return;
    }

    int num_nodes = WU.size();
    for(int node = 0; node < num_nodes; node++) {
        float w = -1.0;
        if(visited[node] == 0) {
            if(source > node) {
                w = WU[node][source - node];
            }
            else if(source < node) {
                w = WU[source][node - source];
            }
            if(w >= 0) {
                weights.push_back(weights[weights.size() - 1] + w);
                dfs_to_target(node, target, WU, visited, min_edge_weights, path, weights);
                visited[node] = 0;
                weights.pop_back();
                path.pop_back();
            }
        }
    }
    return;
}

void weights_dfs(int source, int target, const std::vector<std::vector<float> >& WU, w_table& min_edge_weights)
{
    int num_nodes = WU.size();
    std::vector<int> path;
    std::vector<float> weights;
    std::vector<short> visited (num_nodes, 0);

    weights.push_back(0.0);
    dfs_to_target(source, target, WU, visited, min_edge_weights, path, weights);
}

w_table _get_min_edge_weights(const std::vector<std::vector<float> >& WU, int target)
{
    int num_nodes = WU.size();
    w_table min_edge_weights;
    for(int node_a = 0; node_a < num_nodes; node_a++) {
        for(int node_b = node_a; node_b < num_nodes; node_b++) {
            float w = WU[node_a][node_b];
            if(w >= 0) {
                int edge[2] = {node_a, node_b};
                min_edge_weights[edge_to_str(edge)] = std::numeric_limits<float>::infinity();
            }
        }
    }

    // for(int source = 0; source < num_nodes; source++) {
    for(int source = 0; source < 1; source++) {
        if(source != target) {
            weights_dfs(source, target, WU, min_edge_weights);
            // for(auto e: paths_from_source) {
            //     for(auto j: e) {
            //         std::cout << j << ",";
            //     }
            //     std::cout << std::endl;
            // }
        }
    }

    return min_edge_weights;
}
