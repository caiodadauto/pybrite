import os
import argparse
from tqdm import tqdm

import numpy as np
import networkx as nx

from pytop.utils import add_ip
from pytop.draw import draw_ip_clusters


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", type=str, help="Path of the dataset")
    p.add_argument("--size", default=-1, type=str, help="Size of the dataset")
    p.add_argument("--draw", action="store_true")
    args = p.parse_args()

    random_state = np.random.RandomState(12345)
    if args.size < 0:
        names = [p.split(".")[0] for p in os.listdir(args.path) if p.split(".")[-1] == "gpickle"]
    else:
        names = range(args.size)
    for name in tqdm(names):
        path = os.path.join(args.path, "{}.gpickle".format(name))
        digraph = nx.read_gpickle(path)
        add_ip(digraph, random_state)
        if args.draw:
            draw_ip_clusters(digraph, ext="png", name=name)
        nx.write_gpickle(digraph, path)
