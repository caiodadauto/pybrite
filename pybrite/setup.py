import os
import subprocess as sub
from pathlib import Path
from setuptools import setup
from setuptools.command.install import install
from distutils.command.build import build


BASE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
BRITE_PATH = BASE_PATH.joinpath('brite_cpp')
BIN_PATH = Path.home().joinpath('.local/bin')

if not BIN_PATH.exists():
    os.makedirs(BIN_PATH)

class BriteBuild(build):
    def run(self):
        # run original build code
        build.run(self)

        cmd_make = ['make']
        cmd_bin = ['mv', 'cppgen', str(BIN_PATH)]

        def compile():
            sub.run(cmd_make, cwd=str(BRITE_PATH))
            sub.run(cmd_bin, cwd=str(BRITE_PATH))

        self.execute(compile, [], 'Compiling brite')
        self.mkpath(self.build_lib)

class BriteInstall(install):
    def initialize_options(self):
        install.initialize_options(self)
        self.build_scripts = None

    def finalize_options(self):
        install.finalize_options(self)
        self.set_undefined_options('build', ('build_scripts', 'build_scripts'))

    def run(self):
        # run original install code
        install.run(self)
        self.copy_tree(self.build_lib, self.install_lib)

# def read(fname):
    # return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='pybrite',
    version='0.1',
    description='Topology generator based on Brite',
    maintainer='Caio Dadauto',
    maintainer_email='caio.dadauto@ic.unicamp.br',
    license='GPLv2',
    packages=['pybrite'],
    package_data={
        'pybrite': ['config/*']
    },
    # long_description=read('README.rst'),
    cmdclass={
        'build': BriteBuild,
        'install': BriteInstall
    }
)
