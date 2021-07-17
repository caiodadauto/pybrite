# distutils: language = c++
# distutils: sources = _paths.cpp

from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map


cdef extern from "_min_weights.h":
    unordered_map[string, float] _get_min_edge_weights(vector[vector[float]] WU, int target)

def get_min_edge_weights(object g, int target): #, int n_threads):
    cdef int num_of_nodes = g.number_of_nodes()
    cdef vector[vector[float]] WU
    cdef vector[float] row
    edge_distances = g.edges(data="distance")

    for i in range(num_of_nodes):
        for j in range(i, num_of_nodes):
            row.push_back(-1.0)
        if row.size() > 0:
            WU.push_back(row)
            row.erase(row.begin(), row.end())
    for ep, es, d in edge_distances:
        p = int(ep)
        s = int(es)
        if p > s:
            WU[s][p - s] = float(d)
        else:
            WU[p][s - p] = float(d)

    print(_get_min_edge_weights(WU, target))
