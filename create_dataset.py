import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle

import pytop as pt


def interval(s):
    try:
        min_, max_ = map(int, s.split(","))
        return min_, max_
    except:
        raise argparse.ArgumentTypeError("Interval must be min,max")

def class_ratio(s):
    try:
        h, l = map(float, s.split(","))
        return h, l
    except:
        raise argparse.ArgumentTypeError("Interval must be h,l")

def save_figs(data, params, limits=None):
    sns.set_style("ticks")
    for x, y, name in params:
        fig = plt.figure(dpi=400)
        ax = fig.subplots(1, 1, sharey=False)
        min_x = np.floor(data[x].values.min() *
                         1.05) if data[x].values.min() < 0 else 0
        min_y = np.floor(data[y].values.min() *
                         1.05) if data[y].values.min() < 0 else 0
        max_x = np.ceil(data[x].values.max() *
                        1.05) if data[x].values.max() > 0 else 0
        max_y = np.ceil(data[y].values.max() *
                        1.05) if data[y].values.max() > 0 else 0
        sns.scatterplot(
            x=x,
            y=y,
            data=data,
            style="from",
            size="from",
            hue="from",
            ax=ax,
            legend="full",
            zorder=1,
            sizes=[50],
            markers=["o"],
            palette="Set1",
        )
        # sns.scatterplot(
        #     x=x,
        #     y=y,
        #     data=data,
        #     style="from",
        #     size="from",
        #     hue="from",
        #     ax=ax,
        #     legend="full",
        #     zorder=1,
        #     sizes=[50, 80],
        #     markers=["o", "^"],
        #     palette="Set1",
        # )
        ax.set_xlabel(x, fontsize=18)
        ax.set_ylabel(y, fontsize=18)
        classes = data["Topology Class"].unique().tolist()
        if limits:
            if "Star or H&S" in classes:
                x = np.arange(0, limits[0], limits[0] / 100)
                if limits[1]:
                    y = limits[1]
                else:
                    y = max_y
                _ = ax.add_patch(
                    Rectangle(
                        (0, 0),
                        limits[0],
                        y,
                        ec="k",
                        ls="-",
                        fc=(1, 1, 1, 0),
                        hatch="//",
                        label="Ladder",
                        alpha=0.55,
                    )
                )
                handles, labels = ax.get_legend_handles_labels()
                # _ = handles.pop(labels.index("from"))
                # labels.remove("from")
                ax.legend(
                    handles=handles, loc="lower right", labels=labels, prop={"size": 14}
                )  # data["from"].unique().tolist())
            else:
                x = limits[0]
                if limits[1]:
                    y = limits[1]
                else:
                    y = max_y
                handles, labels = ax.get_legend_handles_labels()
                # _ = handles.pop(labels.index("from"))
                # labels.remove("from")
                ax.legend(
                    handles=handles,
                    loc="lower right",
                    labels=labels,
                    prop={"size": 14},
                )
                ax.axvline(x=x, color="k", lw=1)
                ax.annotate(
                    "",
                    xy=(x * 0.68, y * 0.9),
                    xytext=(x, y * 0.9),
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3", color="k"),
                )
                ax.annotate(
                    "",
                    xy=(x, y * 0.10),
                    xytext=(x * 1.32, y * 0.10),
                    arrowprops=dict(arrowstyle="<-",
                                    connectionstyle="arc3", color="k"),
                )
                plt.text(
                    x * 0.68,
                    y * 0.9,
                    "Star",
                    {
                        "color": "black",
                        "fontsize": 12,
                        "ha": "center",
                        "va": "center",
                        "bbox": dict(boxstyle="round", fc="white", ec="black", pad=0.2),
                    },
                )
                plt.text(
                    x * 1.32,
                    y * 0.10,
                    "H&S",
                    {
                        "color": "black",
                        "fontsize": 12,
                        "ha": "center",
                        "va": "center",
                        "bbox": dict(boxstyle="round", fc="white", ec="black", pad=0.2),
                    },
                )
                ax.plot(
                    [x],
                    [y * 0.10],
                    marker="o",
                    color="k",
                    mec="k",
                    mew=1.2,
                    ls="",
                    zorder=20,
                )
                ax.plot(
                    [x],
                    [y * 0.9],
                    marker="o",
                    color="k",
                    mfc=(1, 1, 1),
                    mec="k",
                    mew=1.2,
                    ls="",
                    zorder=20,
                )
        else:
            handles, labels = ax.get_legend_handles_labels()
            # _ = handles.pop(labels.index("from"))
            # labels.remove("from")
            ax.legend(
                handles=handles, loc="lower right", labels=labels, prop={"size": 14}
            )
        ax.set_xticks(np.arange(min_x, max_x, (max_x - min_x) / 5))
        ax.set_yticks(np.arange(min_y, max_y, (max_y - min_y) / 5))
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
        fig.tight_layout()

        plt.savefig(name, transparent=True)
        fig.clear()
        plt.close()


def get_stats(path, path_stats, imext):
    path_stats = Path(path_stats)
    path = Path(path)
    ladder_dir = path.joinpath("Ladder/")
    star_dir = path.joinpath("Star/")
    hs_dir = path.joinpath("H&S/")
    for d in [ladder_dir, star_dir, hs_dir]:
        os.makedirs(d, exist_ok=True)

    graph_paths = list(path.glob("All/*.gpickle"))
    print("Generate Stats...")
    star_hs = []
    data_stats = np.zeros(
        (len(graph_paths),),
        dtype=[
            ("Number of Nodes", "i4"),
            (r"Average Degree $\bar{\kappa}$", "f8"),
            (r"Maximum Degree $\mu$", "f8"),
            (r"Normalized Maximum Degree $\hat{\mu}$", "f8"),
            (r"High-Degree Ratio $\eta$", "f8"),
            ("Topology Class", "U12"),
            ("from", "U5"),
        ],
    )

    num_hs = 0
    num_star = 0
    num_ladder = 0
    for i in tqdm(range(len(graph_paths))):
        f = graph_paths[i]
        diG = nx.read_gpickle(str(f))
        G = diG.to_undirected()
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        degree = np.array(list(dict(nx.degree(G)).values()))
        avg_degree = 2 * n_edges / n_nodes
        max_degree = degree.max()
        max_degree_norm = max_degree / n_nodes
        R = (degree > avg_degree).sum() / n_nodes
        top_class = (
            "Ladder" if max_degree_norm < 0.4 and avg_degree < 3 else "Star or H&S"
        )
        if top_class == "Star or H&S":
            star_hs.append("Star" if R < 0.25 else "H&S")

        if top_class == "Ladder":
            num_ladder += 1
        elif star_hs[-1] == "Star":
            num_star += 1
        elif star_hs[-1] == "H&S":
            num_hs += 1
        else:
            exit(1)
        if top_class == "Ladder":
            try:
                os.symlink(os.path.abspath(f), os.path.abspath(
                    ladder_dir.joinpath(f.name)))
            except FileExistsError:
                pass
        elif star_hs[-1] == "Star":
            try:
                os.symlink(os.path.abspath(f), os.path.abspath(
                    star_dir.joinpath(f.name)))
            except FileExistsError:
                pass
        else:
            try:
                os.symlink(os.path.abspath(f), os.path.abspath(
                    hs_dir.joinpath(f.name)))
            except FileExistsError:
                pass

        data_stats[i] = (
            n_nodes,
            avg_degree,
            max_degree,
            max_degree_norm,
            R,
            top_class,
            G.graph["from"],
        )
    df_stats = pd.DataFrame(data_stats)

    print("Ladder:", num_ladder, "Star:", num_star, "H&S:", num_hs)

    df_stats.sort_values(by=["from"], inplace=True)
    save_figs(
        df_stats,
        [
            (
                r"Average Degree $\bar{\kappa}$",
                r"Normalized Maximum Degree $\hat{\mu}$",
                path_stats.joinpath("norm_avg_{}.{}".format(path.name, imext)),
            )
        ],
        limits=[3, 0.4],
    )

    non_ladder = df_stats.loc[df_stats["Topology Class"]
                              == "Star or H&S"].copy()
    non_ladder.loc[:, "Topology Class"] = np.array(star_hs, dtype="U12")
    if not non_ladder.empty:
        non_ladder.sort_values(by=["from"], inplace=True)
        save_figs(
            non_ladder,
            [
                (
                    r"High-Degree Ratio $\eta$",
                    r"Maximum Degree $\mu$",
                    path_stats.joinpath("r_{}.{}".format(path.name, imext)),
                )
            ],
            limits=[0.25, None],
        )
    print("Done.")


def clean_dir(path):
    path = Path(path)
    ladder_dir = path.joinpath("Ladder/")
    star_dir = path.joinpath("Star/")
    hs_dir = path.joinpath("H&S/")
    for d in [ladder_dir, star_dir, hs_dir]:
        if os.path.isdir(d):
            for p in list(d.glob("*.gpickle")):
                os.unlink(p)
    for p in list(path.glob("All/*.gpickle")):
        os.remove(p)


def get_last(path):
    path = Path(path)
    ladder_dir = path.joinpath("Ladder/")
    star_dir = path.joinpath("Star/")
    hs_dir = path.joinpath("H&S/")
    for d in [ladder_dir, star_dir, hs_dir]:
        if os.path.isdir(d):
            for p in d.glob("*.gpickle"):
                os.rename(p, path.joinpath(p.name))
    sorted_list = sorted(path.glob("*.gpickle"), key=lambda x: int(x.stem))
    if sorted_list:
        return int(sorted_list[-1].stem) + 1
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("size", type=int, help="Number of graphs")
    p.add_argument(
        "--type",
        type=str,
        default="brite",
        choices=["zoo", "brite"],
        help="Get graphs from brite or topology zoo",
    )
    p.add_argument(
        "--raw-zoo-dir", type=str, default="zoo/", help="Path to gml topology zoo file"
    )
    p.add_argument("--path", type=str, default="dataset/",
                   help="Path to save data")
    p.add_argument(
        "--n-interval",
        type=interval,
        default=(25, 55),
        help="Interval for number of nodes.",
    )
    p.add_argument(
        "--composition-interval",
        type=interval,
        default=(1, 4),
        help="Interval for the number of compositions used to create the final graph",
    )
    # p.add_argument(
    #     "--max-m",
    #     type=interval,
    #     default=4,
    #     help="The maximum number of edges to be added for each new nodes.",
    # )
    p.add_argument(
        "--plane-size",
        type=int,
        default=1200,
        help="The plane size where the nodes and edges will be added",
    )
    p.add_argument("--test", action="store_true", help="Save data for test")
    p.add_argument(
        "--generalization",
        action="store_true",
        help="Save data for test in the generalization directory",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Append to a dataset already existent"
    )
    p.add_argument(
        "--only-stats",
        action="store_true",
        help="Create statstics from a given dataset",
    )
    p.add_argument(
        "--imext",
        type=str,
        default="pdf",
        choices=["pdf", "jpg", "png", "svg"],
        help="Image extension for statstics",
    )
    p.add_argument(
        "--non-balanced",
        action="store_true",
        help="Not require that the dataset will be balanced in the number of"
        " topologies for each class",
    )
    p.add_argument(
        "--class-ratio",
        type=class_ratio,
        default=(1/3, 1/3),
        help="Proportion of Ladder and H&S topologies",
    )
    args = p.parse_args()

    ratio = args.n_interval[0] // args.composition_interval[1]
    if ratio < 6 and args.type == "brite":
        raise ValueError(
            "The minumum value riquered for number of nodes"
            " is equal to the maximum number of compositions times 6."
        )

    if args.test:
        if args.generalization:
            path = os.path.join(args.path, "test_generalization")
        else:
            path = os.path.join(args.path, "test_non_generalization")
    else:
        path = os.path.join(args.path, "train")
    # path = os.path.join(args.path, "data_to_plot")
    all_path = os.path.join(path, "All")
    path_stats = os.path.join(args.path, "stats")
    os.makedirs(path_stats, exist_ok=True)

    if not args.only_stats:
        if not (os.path.isdir(all_path)):
            os.makedirs(all_path)
            offset = 0
        else:
            if args.overwrite:
                go = input(
                    "The data will be overwrite, do you want to continue? [yN]")
                if go == "y":
                    clean_dir(path)
                    offset = 0
                else:
                    print("Abort the data creation.")
                    exit(0)
            else:
                go = input(
                    "The new data will be appended to the existingdata,"
                    " do you want to continue? [yN]"
                )
                if go == "y":
                    clean_dir(path)
                    offset = get_last(path)
                else:
                    print("Abort the data creation.")
                    exit(0)

        if args.type == "zoo":
            try:
                assert os.path.isdir(args.raw_zoo_dir)
            except AssertionError:
                print("Path {} is not a valid diretory.".format(args.raw_zoo_dir))
                exit(1)
            print("For zoo, parameter 'size' will be ignored")
            pt.create_static_zoo_dataset(
                os.path.abspath(all_path),
                os.path.abspath(args.raw_zoo_dir),
                args.n_interval,
                offset=offset,
            )
        else:
            pt.create_static_brite_dataset(
                os.path.abspath(all_path),
                args.size,
                args.n_interval,
                args.composition_interval,
                class_ratio=args.class_ratio,
                main_plane_size=args.plane_size,
                offset=offset,
                balanced=False if args.non_balanced else True,
            )
    get_stats(path, path_stats, args.imext)
