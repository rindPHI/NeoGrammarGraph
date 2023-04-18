# NeoGrammarGraph

Creating graphs from context-free grammars for fun and profit.

This project is a re-implementation of [GrammarGraph](https://github.com/rindPHI/GrammarGraph/)
based on the efficient [graph-tool](https://graph-tool.skewed.de/) library. As of now,
it does not support the full set of features supported by GrammarGraph; however, all
implemented features should work significantly more efficiently thanks to graph-tool.

## Supported Features

* Reachability
* Export to GraphViz DOT files.

## Planned Features

* Creating sub graphs
* Filter abstraction
* Dijkstra's algorithm for shortest paths between nodes
* Checking whether a (sub) graph represents a tree
* Computing k-paths (paths of exactly length k) in grammars and derivation trees, and a 
  k-path coverage measure ([see this paper](https://ieeexplore.ieee.org/document/8952419)) of 
  derivation trees based on that.

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

Author: [Dominic Steinhöfel](https://www.dominic-steinhoefel.de).
