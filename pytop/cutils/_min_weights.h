#ifndef _PATH_H
#define _PATH_H

#include<vector>
#include <string>
#include <unordered_map>

typedef std::unordered_map<std::string, float> w_table;

std::string edge_to_str(int a[2]);
void dfs_to_target(int source, int target, const std::vector<std::vector<float> >& WU, std::vector<short>& visited, w_table& min_edge_weights, std::vector<int>& path, std::vector<float>& weights);
void weights_dfs(int source, int target, const std::vector<std::vector<float> >& WU, w_table& min_edge_weights);
w_table _get_min_edge_weights(const std::vector<std::vector<float> >& WU, int target);

#endif
