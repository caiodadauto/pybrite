# Pytop
![](https://github.com/caiodadauto/pytop/blob/master/graphs.png)

**Pytop** is a Python package to generate topologies based on the *Boston University Representative Internet Topology Generator (BRITE)* and the dataset
[Topology Zoo](http://www.topology-zoo.org/). This interface
intend to create batches generator of topologies to be inputed in a deep learning model for graphs.

## Installation

Clone project, go to the `pytop` directory and make
```
pip install --user .
```
The `pip` will compile the *BRITE* and move the binary to the `$HOME/.local/bin`; so, please, verify if this location is in the `PATH`
environment variable.

## Usage

Basically, `pytop` translates the topologies created by *BRITE* and available in Topology Zoo.
All graphs delivery by `pytop` have two associated properties, one for vertices (called *pos*, which is the position) and
other for edges (called *weight*, which is the euclidean distance between vertices). The usage example is presented in
this [jupyter notebook](https://github.com/caiodadauto/pytop/blob/master/Usage.ipynb).

## Credits

The core of the `pytop` was implemented by *Alberto Medina* and *Anukool Lakhina* from *Boston University*. The original documentation
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
