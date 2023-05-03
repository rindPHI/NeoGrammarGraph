# NeoGrammarGraph

[![Python](https://img.shields.io/pypi/pyversions/neo-grammar-graph.svg)](https://pypi.python.org/pypi/neo-grammar-graph/)
[![Version](https://img.shields.io/pypi/v/neo-grammar-graph)](https://pypi.python.org/pypi/neo-grammar-graph/)
[![BuildStatus](https://img.shields.io/github/actions/workflow/status/rindPHI/NeoGrammarGraph/test-gg.yml?branch=main)](https://img.shields.io/github/actions/workflow/status/rindPHI/NeoGrammarGraph/test-gg.yml?branch=main)
[![Coverage Status](https://coveralls.io/repos/github/rindPHI/NeoGrammarGraph/badge.svg?branch=main)](https://coveralls.io/github/rindPHI/NeoGrammarGraph?branch=main)
[![Dependencies](https://img.shields.io/librariesio/release/github/rindphi/NeoGrammarGraph)](https://libraries.io/github/rindPHI/NeoGrammarGraph)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Creating graphs from context-free grammars for fun and profit.

This project is a re-implementation of [GrammarGraph](https://github.com/rindPHI/GrammarGraph/)
based on the efficient [graph-tool](https://graph-tool.skewed.de/) library.

## Supported Features

* Creating sub graphs
* Filter abstraction
* Reachability of grammar symbols.
* Computing the shortest path or all paths between two grammar symbols.
* Computing k-paths (paths of exactly length k) in grammars and derivation trees 
  ([see this paper](https://ieeexplore.ieee.org/document/8952419)).
* Checking whether a (sub) graph represents a tree
* Export to GraphViz DOT files.

## Install

NeoGrammarGraph requires at least Python 3.10 and a working installation of graph-tool.
We refer to the graph-tool homepage (https://graph-tool.skewed.de/) for instructions
on how to obtain graph-tool. On a MacOS system, we recommend the installation using
`homebrew`; on Debian/Ubuntu, there's a custom APT repository to be used with `apt-get`.

We recommend installing NeoGrammarGraph in a virtual environment.
Example usage (inside project directory):

```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Run tests
python3 -m pytest
```

On MacOS (maybe also on other systems), it might be required to link the graph-tool
library into the virtual environment. If you used `homebrew` to install graph-tool,
an example command-line invocation could look as follows (you might have to update
some paths depending on the Python/graph-tool versions on your system):

```shell
ln -s \
    /usr/local/Cellar/graph-tool/2.51/lib/python3.11/site-packages/graph_tool \
    venv/lib/python3.10/site-packages
```

For the GitHub workflow, the following line was required:

```shell
sudo ln -s \
    /usr/lib/python3/dist-packages/graph_tool \
    /opt/hostedtoolcache/Python/3.10.11/x64/lib/python3.10/site-packages
```

Author: [Dominic Steinhöfel](https://www.dominic-steinhoefel.de).
