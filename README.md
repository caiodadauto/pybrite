# Pybrite


----------------

## What is it?
**Pybrite** is a Python interface to the *Boston University Representative Internet Topology Generator (BRITE)*; this interface
intend to create a batch generator of *N* topologies, which is easily integrate with a graph deep learning models written in Python.

## Installation
Clone project, go to the `pybrite` directory and make
```
pip install --user .
```
The `pip` will compile the *BRITE* and move the binary to the `$HOME/.local/bin`; so, please, verify if this location is in the `PATH`
environment variable.

## Usage
Basically, `pybrite` translate the topologies created by *BRITE* to the undirected graph structure defined by `graph_tool` package.
All graphs delivery by `pybrite` have two associated properties, one for vertices (called *pos*, which is the position) and
other for edges (called *weight*, which is the distance between vertices). The usage example is presented in
this [jupyter notebook](https://github.com/caiodadauto/pybrite/blob/master/Usage.ipynb).

## Credits
The core of the `pybrite` was implemented by *Alberto Medina* and *Anukool Lakhina* from *Boston University*. The original documentation
for this project can be found [here](https://www.cs.bu.edu/brite/index.html).

## Licence
The core follow copyright from *Boston University*, which can be read below

>                  Copyright 2001, Trustees of Boston University.
>                               All Rights Reserved.
>
> Permission to use, copy, or modify this software and its documentation
> for educational and research purposes only and without fee is hereby
> granted, provided that this copyright notice appear on all copies and
> supporting documentation.  For any other uses of this software, in
> original or modified form, including but not limited to distribution in
> whole or in part, specific prior permission must be obtained from Boston
> University.  These programs shall not be used, rewritten, or adapted as
> the basis of a commercial software or hardware product without first
> obtaining appropriate licenses from Boston University.  Boston University
> and the author(s) make no representations about the suitability of this
> software for any purpose.  It is provided "as is" without express or
> implied warranty.
