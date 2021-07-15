import os
import subprocess as sub
from pathlib import Path
from setuptools import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError("Cython is not installed")


def build(setup_kwargs):
    def build_brite():
        base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        brite_path = base_path.joinpath("external/brite_cpp")
        bin_path = Path("/usr/local/bin")
        if not os.access(bin_path, os.W_OK):
            bin_path = Path.home().joinpath(".local/bin")
        if not bin_path.exists():
            os.makedirs(bin_path)

        if not os.path.isfile(bin_path.joinpath("cppgen")):
            cmd_make = ["make"]
            cmd_bin = ["mv", "cppgen", str(bin_path)]
            sub.run(cmd_make, cwd=str(brite_path))
            sub.run(cmd_bin, cwd=str(brite_path))

    # ext = [
    #     Extension(
    #         name="itdk.graph_utils.paths",
    #         sources=["itdk/graph_utils/_paths.cpp", "itdk/graph_utils/paths.pyx"],
    #         extra_compile_args=["-fopenmp"],
    #         extra_link_args=["-fopenmp"],
    #         language="c++",
    #     )
    # ]
    # setup_kwargs.update({"ext_modules": ext, "cmdclass": {"build_ext": build_ext}})
    build_brite()
