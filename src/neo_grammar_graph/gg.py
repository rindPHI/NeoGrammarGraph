# Copyright © 2023 CISPA Helmholtz Center for Information Security.
# Author: Dominic Steinhöfel.
#
# This file is part of NeoGrammarGraph.
#
# NeoGrammarGraph is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# NeoGrammarGraph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NeoGrammarGraph.  If not, see <http://www.gnu.org/licenses/>.
import os.path
from functools import cache
from typing import Dict, List, Callable, Optional, Tuple, Any, cast, Sequence

import numpy as np
from bidict import MutableBidict, bidict
from cachetools import cached
from cachetools.keys import hashkey
from graph_tool import Graph, Vertex, Edge, GraphView
from graph_tool.search import bfs_search, BFSVisitor, StopSearch
from graph_tool.spectral import adjacency
from graph_tool.topology import transitive_closure, all_paths
from orderedset import OrderedSet, FrozenOrderedSet

from neo_grammar_graph.helpers import split_expansion, is_nonterminal, canonical
from neo_grammar_graph.nodes import (
    ChoiceNode,
    Node,
    NonterminalNode,
    TerminalNode,
)
from neo_grammar_graph.type_defs import Grammar, ParseTree, CanonicalGrammar


class NeoGrammarGraph:
    def __init__(
        self,
        grammar: Grammar,
    ):
        """
        Constructs a :class:`~neo_grammar_graph.NeoGrammarGraph` from the given
        :class:`~neo_grammar_graph.type_defs.Grammar`. The graph is constructed
        roughly according to the descriptions in the following paper:

            Nikolas Havrikov, Andreas Zeller:
            Systematically Covering Input Structure. ASE 2019: 189-199
            https://doi.org/10.1109/ASE.2019.00027

        However, our grammars are simple BNF, not EBNF grammars. Consequently, there
        are is no star/plus operator etc. Furthermore, we do not use concatenation
        nodes, and the alternation nodes from Havrikov et al. are called "choice nodes"
        in our implementation.

        In the documentation of this class, we frequently use the following example
        grammar:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }

        You should get a feeling about the structure of the graphs produced by this
        class when having a look at the graphical representation of this grammar.

        >>> graph = NeoGrammarGraph(grammar)

        >>> import tempfile
        >>> fp = tempfile.NamedTemporaryFile(suffix=".dot")
        >>> graph.save_to_dot(fp.name)
        >>> # print(fp.name)  # <-- run `dot -Tpng {fp.name} -o {fp.name}.png`
        >>>                   #     and open `{fp.name}.png` to inspect the result.

        NeoGrammarGraph uses the `graph-tool <https://graph-tool.skewed.de/>`_
        library for efficient graph operations such as search. A working graph-tool
        installation (native, as graph-tool is implemented as a native libray) is
        required to use this module.

        A :class:`~neo_grammar_graph.gg.NeoGrammarGraph` object stores the "canonical"
        representation of the given grammar in the field :code:`canonical_grammar` for
        convenience reasons of certain applications. Expansion alternatives in
        canonical grammars contain individual (non-)terminal symbols in lists:

        >>> graph.grammar["<stmt>"]
        ['<assgn> ; <stmt>', '<assgn>']

        >>> graph.canonical_grammar["<stmt>"]
        [['<assgn>', ' ; ', '<stmt>'], ['<assgn>']]

        :param grammar: The grammar to construct the
            :class:`~neo_grammar_graph.NeoGrammarGraph` object from. The grammar must
            contain a nonterminal :code:`<start>` that is interpreted as the start
            symbol of the grammar.
        """

        self.grammar: Grammar = grammar
        self.__canonical_grammar: Optional[CanonicalGrammar] = None
        self.graph: Graph = Graph()
        self.closure: Optional[Graph] = None
        self.vertex_to_node: MutableBidict[Vertex, Node] = bidict({})
        self.symbol_to_vertices: Dict[str, OrderedSet[Vertex]] = {}

        # Cache
        self.__hash: Optional[int] = None

        self.__initialize_graph()

    @property
    def canonical_grammar(self) -> CanonicalGrammar:
        if self.__canonical_grammar is None:
            self.__canonical_grammar = canonical(self.grammar)

        return self.__canonical_grammar

    def __initialize_graph(self) -> None:
        """
        Initializes the graph: Adds vertices and edges according to the grammar,
        registers choice nodes, sets the vertex labels to the corresponding grammar
        symbols, and populates the map from grammar symbols to vertices.

        :return: Nothing. Writes to :code:`self.vertex_to_node`,
            :code:`self.symbol_to_vertices`, and resets & populates :code:`self.graph`.
        """

        start_vertex = self.graph.add_vertex()
        start_node = NonterminalNode(0, "<start>")

        self.vertex_to_node: MutableBidict[Vertex, Node] = bidict(
            {start_vertex: start_node}
        )

        self.symbol_to_vertices: Dict[str, OrderedSet[Vertex]] = {
            "<start>": OrderedSet({start_vertex})
        }

        # This property primarily serves to convey the children order in DOT
        # graph exports.
        self.graph.ep.label = self.graph.new_edge_property("int")

        # We only consider reachable nodes by adding nonterminals to the stack
        # when they're first discovered.
        stack = [start_node]
        while stack:
            nonterminal_node = stack.pop(0)
            assert nonterminal_node in self.vertex_to_node.inverse
            assert nonterminal_node.value in self.symbol_to_vertices

            for alternative_nr, alternative in enumerate(
                self.grammar[nonterminal_node.value]
            ):
                choice_node = ChoiceNode(alternative_nr, nonterminal_node.value)

                choice_vertex = self.graph.add_vertex()
                self.vertex_to_node[choice_vertex] = choice_node

                nonterminal_choice_edge = self.graph.add_edge(
                    self.vertex_to_node.inverse[nonterminal_node], choice_vertex
                )
                self.graph.ep.label[nonterminal_choice_edge] = alternative_nr

                # If `alternative` is the empty expansion (epsilon), we still have to
                # ensure that an epsilon terminal node is added even though
                # `split_expansion` would return an empty list. Thus, we keep a custom
                # singleton list containing an empty string in that case.
                for elem_nr, expansion_element in enumerate(
                    split_expansion(alternative) or [""]
                ):
                    ident = len(
                        self.symbol_to_vertices.setdefault(
                            expansion_element, OrderedSet()
                        )
                    )

                    child_node = (
                        NonterminalNode(ident, expansion_element)
                        if is_nonterminal(expansion_element)
                        else TerminalNode(ident, expansion_element)
                    )
                    child_vertex = self.graph.add_vertex()

                    self.symbol_to_vertices[expansion_element].add(child_vertex)
                    self.vertex_to_node[child_vertex] = child_node

                    choice_expansion_elem_edge = self.graph.add_edge(
                        choice_vertex, child_vertex
                    )
                    self.graph.ep.label[choice_expansion_elem_edge] = elem_nr

                    if isinstance(child_node, NonterminalNode) and ident == 0:
                        stack.append(child_node)

        self.graph.vp.label = self.graph.new_vertex_property("string")
        for vertex, node in self.vertex_to_node.items():
            self.graph.vp.label[vertex] = str(node)

        # Add edges from "reference" nonterminal nodes to the children of the
        # primary NonterminalNode for that nonterminal. The background is that
        # we create individual nodes for each occurrence of a nonterminal in
        # any expansion alternative. The first one already contains the children
        # as defined by the grammar; the subsequent ones do not have any outgoing
        # edges yet.

        for vertices in self.symbol_to_vertices.values():
            if len(vertices) < 2:
                # No reference nodes
                continue

            primary_vertex = vertices[0]
            reference_vertices = vertices[1:]

            for reference_vertex in reference_vertices:
                for child_nr, child_vertex in enumerate(primary_vertex.out_neighbors()):
                    edge = self.graph.add_edge(reference_vertex, child_vertex)
                    self.graph.ep.label[edge] = child_nr

    def __eq__(self, other: Any) -> bool:
        """
        Returns True if the other object is a NeoGrammarGraph object with the same
        grammar.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        A graph equals itself.

        >>> graph == graph
        True

        Two different graph objects are equal if they have the same grammar.

        >>> NeoGrammarGraph(grammar) == NeoGrammarGraph(grammar)
        True

        But it is different from a proper subgraph of itself.

        >>> NeoGrammarGraph(grammar) == NeoGrammarGraph(grammar).subgraph("<assgn>")
        False

        :param other: The object to compare.
        :return: True if the other object is a NeoGrammarGraph object with the same
            grammar.
        """

        return isinstance(other, NeoGrammarGraph) and self.grammar == other.grammar

    def __hash__(self):
        """
        Computes the hash of this grammar graph object based on the grammar. The hash
        is cached.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The hashes of graph objects with the same grammar are equal.

        >>> hash(NeoGrammarGraph(grammar)) == hash(NeoGrammarGraph(grammar))
        True

        The hash of a sub graph is (in this example!) different.

        >>> hash(NeoGrammarGraph(grammar)) == \
                hash(NeoGrammarGraph(grammar).subgraph("<assgn>"))
        False

        The sub graph for :code:`<start>` is the same as the original graph, so the
        hashes are also identical.

        >>> hash(NeoGrammarGraph(grammar)) == \
                hash(NeoGrammarGraph(grammar).subgraph("<start>"))
        True

        :return: A hash value for this graph.
        """

        if self.__hash is None:
            self.__hash = hash(
                tuple(
                    [
                        (nonterminal, tuple(expansions))
                        for nonterminal, expansions in self.grammar.items()
                    ]
                )
            )

        return self.__hash

    def vertex_indices(self, symbol: str | Node) -> Optional[OrderedSet[int]]:
        """
        Returns the vertex indices for the given grammar symbol.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> print(graph.vertex_indices("<start>"))
        {0}

        >>> print(graph.vertex_indices("<stmt>"))
        {2, 6}

        >>> print(graph.vertices(NonterminalNode(0, "<stmt>")))
        {2}

        >>> print(graph.vertices(NonterminalNode(1, "<stmt>")))
        {6}

        >>> print(graph.vertex_indices("a"))
        {14}

        >>> graph.vertex_indices("<function>") is None
        True

        :param symbol: The grammar symbol.
        :return: The index of the given grammar symbol in the graph, or None if the
            symbol does not exist in the graph.
        """

        vertices = self.vertices(symbol)
        if vertices is None:
            return None

        return OrderedSet([self.graph.vertex_index[vertex] for vertex in vertices])

    def vertices(self, symbol: str | Node) -> Optional[OrderedSet[Vertex]]:
        """
        Returns the vertex objects associated to the given grammar symbol.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> print(graph.vertices("<start>"))
        {0}

        >>> print(graph.vertices("<stmt>"))
        {2, 6}

        >>> print(graph.vertices(NonterminalNode(0, "<stmt>")))
        {2}

        >>> print(graph.vertices(NonterminalNode(1, "<stmt>")))
        {6}

        >>> print(graph.vertices("a"))
        {14}

        >>> graph.vertices("<function>") is None
        True

        :param symbol: The grammar symbol.
        :return: The vertex objects for the given grammar symbol in the graph, or None
            if the symbol does not exist in the graph.
        """

        if isinstance(symbol, Node):
            return OrderedSet([self.vertex_to_node.inverse[symbol]])

        return self.symbol_to_vertices.get(symbol, None)

    def nodes(self, symbol: Optional[str] = None) -> Optional[OrderedSet[Node]]:
        """
        Returns a list of all graph nodes, including the artificial choice nodes.
        If a symbol is given, all nodes associated to that symbol are returned.
        If the given symbol is no grammar symbol, None is returned.
        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        Including terminal symbols and choice nodes, there are 86 nodes in the graph:

        >>> len(graph.nodes())
        89

        These are the first ten nodes:

        >>> print(", ".join(map(str, graph.nodes()[:5])))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)

        If a symbol is provided, only the associated nodes are returned:

        >>> print(graph.nodes("<stmt>"))
        {<stmt> (0), <stmt> (1)}

        >>> graph.nodes("<lalala>") is None
        True

        :param symbol: An optional symbol. If provided, the nodes associated to that
            symbol are returned.
        :return: All nodes or the nodes associated to the given symbol in this
            grammar graph.
        """

        if symbol is not None:
            vertices = self.symbol_to_vertices.get(symbol, None)
            if vertices is None:
                return None

            return OrderedSet([self.vertex_to_node[vertex] for vertex in vertices])

        return OrderedSet([self.node(v) for v in self.graph.vertices()])

    def edges(self) -> List[Tuple[Node, Node]]:
        """
        Returns a list of all edges in this graph.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> print("\\n".join(map(
        ...     lambda edge: f"({edge[0]}, {edge[1]})",
        ...     graph.edges()[:10])))
        (<start> (0), <start>-choice (0))
        (<start>-choice (0), <stmt> (0))
        (<stmt> (0), <stmt>-choice (0))
        (<stmt> (0), <stmt>-choice (1))
        (<stmt>-choice (0), <assgn> (0))
        (<stmt>-choice (0), ' ; ' (0))
        (<stmt>-choice (0), <stmt> (1))
        (<assgn> (0), <assgn>-choice (0))
        (<stmt> (1), <stmt>-choice (0))
        (<stmt> (1), <stmt>-choice (1))

        :return: All edges in the grammar graph. Contains edges to artificial
            "choice nodes."
        """

        return [
            (self.node(edge.source()), self.node(edge.target()))
            for edge in self.graph.edges()
        ]

    def node(self, vertex: Vertex) -> Node:
        """
        Returns the :class:`~neo_grammar_graph.nodes.Node` object (either a terminal or
        nonterminal node or a choice node) for the given graph vertex.

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> print(graph.node(graph.graph.vertex(0)))
        <start> (0)

        >>> print(graph.node(graph.graph.vertex(1)))
        <start>-choice (0)

        :param vertex: The vertex for which to return the associated symbol
        :return: The symbol associated to the given vertex.
        """

        return self.vertex_to_node[vertex]

    @cache
    def children(self, node: Node) -> Optional[List[Node]]:
        """
        Returns the immediate children of a Node in the graph. If the given
        Node does not exist in the graph, :code:`None` is returned.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.children(NonterminalNode(0, "<stmt>"))
        [ChoiceNode(ident=0, parent_nonterminal='<stmt>'), ChoiceNode(ident=1, parent_nonterminal='<stmt>')]

        >>> graph.children(ChoiceNode(ident=0, parent_nonterminal='<stmt>'))
        [NonterminalNode(ident=0, value='<assgn>'), TerminalNode(ident=0, value=' ; '), NonterminalNode(ident=1, value='<stmt>')]

        >>> graph.children(TerminalNode(ident=0, value=' ; '))
        []

        >>> graph.children(TerminalNode(ident=0, value=' nope ')) is None
        True

        :param node: The parent node.
        :return: The children of :code:`node` in the graph.
        """

        vertex = self.vertex_to_node.inverse.get(node, None)
        if vertex is None:
            return None

        return [self.node(child_vertex) for child_vertex in vertex.out_neighbors()]

    def subgraph(self, start_nonterminal: str) -> "NeoGrammarGraph":
        r"""
        Computes a sub graph for that part of the grammar starting with
        :code:`start_nonterminal`. The resulting grammar will still start with
        "<start>", but this initial nonterminal will be connected to
        :code:`start_nonterminal`.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The nonterminals of the sub graph correspond to those reachable from the chosen
        start nonterminal. In particular, :code:`<stmt>` is missing in the below
        example:

        >>> subgraph = graph.subgraph("<assgn>")
        >>> list(subgraph.grammar.keys())
        ['<start>', '<assgn>', '<rhs>', '<var>', '<digit>']

        The :code:`<start>` nonterminal is connected to :code:`start_nonterminal`:

        >>> path = [
        ...     (s, t) for s, t in subgraph.edges()
        ...     if (isinstance(s, NonterminalNode) and s.value == "<start>") or
        ...        (isinstance(t, NonterminalNode) and t.value == "<assgn>")
        ... ]
        >>> print("\n".join(map(str, path)))
        (NonterminalNode(ident=0, value='<start>'), ChoiceNode(ident=0, parent_nonterminal='<start>'))
        (ChoiceNode(ident=0, parent_nonterminal='<start>'), NonterminalNode(ident=0, value='<assgn>'))

        The only edge of a sub graph not contained in the original graph is part of
        the connection of :code:`<start>` to the :code:`start_nonterminal` (via the
        corresponding choice node):

        >>> set(subgraph.edges()).difference(set(graph.edges()))
        {(ChoiceNode(ident=0, parent_nonterminal='<start>'), NonterminalNode(ident=0, value='<assgn>'))}

        Two arbitrary nodes in the subgraph are connected if, and only if, they are
        connected in the original graph:

        >>> for n_1 in subgraph.nodes():
        ...     for n_2 in subgraph.nodes():
        ...         assert (
        ...             not graph.reachable(n_1, n_2) and
        ...             not subgraph.reachable(n_1, n_2) or
        ...             graph.reachable(n_1, n_2) and
        ...             subgraph.reachable(n_1, n_2))

        The "sub graph" for the nonterminal symbol :code:`<start>` itself is the
        original graph:

        >>> graph is graph.subgraph("<start>")
        True

        :param start_nonterminal: The nonterminal symbol determining the sub grammar
            of the sub graph to be computed.
        :return: A sub graph for the grammar starting with :code:`start_nonterminal`.
            The resulting graph/grammar satisfy the convention that the grammar starts
            with the initial symbol :code:`<start>`, which is connected to
            :code:`start_nonterminal`.
        """

        # We always assume the start nonterminal of a grammar to be :code:`<start>`.
        # If this nonterminal is given, we return the original NeoGrammarGraph object.
        # Note that we do not copy the object, it will literally be the same one.
        if start_nonterminal == "<start>":
            return self

        # Create the new grammar using graph reachability
        new_grammar = {"<start>": [start_nonterminal]} | {
            nonterminal: list(expansion)
            for nonterminal, expansion in self.grammar.items()
            if nonterminal == start_nonterminal
            or self.reachable(start_nonterminal, nonterminal)
        }

        return NeoGrammarGraph(new_grammar)

    def is_tree(self) -> bool:
        """
        Checks whether this graph is a tree, i.e., each node has exactly one parent.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The assignment language grammar is recursive, so it cannot be a tree.

        >>> graph.is_tree()
        False

        Though assignments are not recursive, they are no trees: The :code:`<var>` node
        is reachable from an assignment or a :code:`<rhs>` nonterminal. The subgraph
        for :code:`<assgn>` is a directed, acyclic graph (DAG).

        >>> graph.subgraph("<assgn>").is_tree()
        False

        >>> graph.subgraph("<rhs>").is_tree()
        True

        :return: True iff this graph is a tree, i.e., each node has at most one parent.
        """

        class TreeVisitor(BFSVisitor):
            def __init__(self):
                self.result = True

            def non_tree_edge(self, e):
                self.result = False
                raise StopSearch()

        v = TreeVisitor()
        bfs_search(self.graph, source=self.vertices("<start>")[0], visitor=v)

        return v.result

    def filter(self, filter_function: Callable[[Node], bool]) -> OrderedSet[Node]:
        """
        Computes the set of graph nodes satisfying the criterion determined by
        :code:`filter_function`.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        Filtering out all choice nodes and all terminal symbols (nodes without children)
        yields all grammar nonterminals.

        >>> print(OrderedSet(map(
        ...     lambda node: node.value,
        ...    graph.filter(lambda node: isinstance(node, NonterminalNode)))))
        {<start>, <stmt>, <assgn>, <var>, <rhs>, <digit>}

        On the other hand, we can also specifically ask for all terminal symbols:

        >>> list(map(
        ...     lambda node: node.value,
        ...     graph.filter(lambda node: isinstance(node, TerminalNode))[:14]))
        [' ; ', ' := ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

        :param filter_function: A function taking a grammar symbol (nonterminals,
            terminals, or choice nodes) and deciding whether it should be filtered out
            (return False) or kept (return True).
        :return: All grammar symbols (potentially including choice nodes) satisfying
            the given filter criterion.
        """

        prop = self.graph.new_vertex_property("bool")

        for vertex in self.graph.vertices():
            prop[vertex] = filter_function(self.node(vertex))

        # Create the new graph-tool graph as a filtered version of the original one.
        return OrderedSet(
            map(self.node, Graph(g=GraphView(self.graph, vfilt=prop)).vertices())
        )

    @cache
    def reachable(self, from_symbol: str | Node, to_symbol: str | Node) -> bool:
        """
        Checks whether the nonterminal symbol :code:`to_nonterminal` is reachable
        in the graph from the nonterminal symbol :code:`from_nonterminal`. Note that
        reachability is not reflexive; a nonterminal is only reachable from itself if
        there is an actual path starting and ending at that nonterminal.

        On the first call to this method, the transitive closure of the grammar graph
        is computed once; for all further invocations, reachablility is computed based
        on the transitive closure, which works in constant time. Since grammar graphs
        are usually not very dense, computing the transitive closure on the first
        invocation is expected to perform bettern than conducting a search from scratch
        on the original graph for each invocation.

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.reachable("<stmt>", "<stmt>")
        True

        >>> graph.reachable("<stmt>", "<assgn>")
        True

        >>> graph.reachable("<assgn>", "<stmt>")
        False

        >>> graph.reachable("<assgn>", "<assgn>")
        False

        If a node does not exist, this method returns False:

        >>> graph.reachable("<does-not-exist>", "<assgn>")
        False

        :param from_symbol: The nonterminal starting from which reachability
            should be checked.
        :param to_symbol: The nonterminal that should be reached.
        :return: True iff :code:`to_nonterminal` is reachable from
            :code:`from_nonterminal`.
        """

        if self.closure is None:
            self.closure = transitive_closure(self.graph)

        start_vertex_indices = self.vertex_indices(from_symbol)
        target_vertices = self.vertices(to_symbol)

        if start_vertex_indices is None or target_vertices is None:
            return False

        # We need to make sure to retrieve the start vertices from the closure,
        # since we use the :code:`out_neighbors()` function on these objects.
        start_vertices = [self.closure.vertex(idx) for idx in start_vertex_indices]

        return any(
            target_vertex in start_vertex.out_neighbors()
            for start_vertex in start_vertices
            for target_vertex in target_vertices
        )

    def shortest_path(
        self,
        source_symbol: str,
        target_symbol: str,
        node_filter: Callable[[Node], bool] = lambda node: not isinstance(
            node, ChoiceNode
        ),
    ) -> Optional[List[Node]]:
        """
        Computest the shortest path between two nonterminals in the grammar graph
        using a breadth-first search. Note that a more general algorithm like
        Dijkstra's is not required here since all edge weights in grammar graphs
        are equal.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The shortest path between :code:`<stmt>` and :code:`<digit>` in the above
        grammar consists of four nonterminals:

        >>> print(path_to_str(graph.shortest_path("<stmt>", "<digit>")))
        <stmt> (0), <assgn> (0), <rhs> (0), <digit> (0)

        If we relax the default :code:`node_filter`, we are also presented the
        "choice nodes:"

        >>> print(path_to_str(
        ...     graph.shortest_path("<stmt>", "<digit>", lambda _: True)))
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        For unconnected nonterminals (grammar graphs are directed!), we obtain an empty
        list.

        >>> graph.shortest_path("<digit>", "<stmt>")
        []

        For nonterminals reachable by themselves, the end node might be a different
        one than the start node (though they share the same nonterminal symbol), since
        the nonterminal is reached via a different context.

        >>> print(path_to_str(
        ...     graph.shortest_path("<stmt>", "<stmt>", lambda _: True)))
        <stmt> (0), <stmt>-choice (0), <stmt> (1)

        Reachability is not reflexive: If there is no path of at least one edge between
        a nonterminal and itself, an empty list is returned.

        >>> graph.shortest_path("<digit>", "<digit>")
        []

        If either the source or the target symbol does not exist, :code:`None` is
        returned.
        >>> graph.shortest_path("<single-digit>", "<digit>") is None
        True

        Nodes in the result can be arbitrarily filtered; let's take out the source
        nonterminal:

        >>> print(path_to_str(graph.shortest_path(
        ...     "<stmt>",
        ...     "<digit>",
        ...     lambda node: (
        ...         not isinstance(node, NonterminalNode) or
        ...         node.value != "<stmt>"))))
        <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        :param source_symbol: The start nonterminal for the computation of a
            shortest path.
        :param target_symbol: The destination nonterminal for the computation of
            a shortest path.
        :param node_filter: A function for filtering out elements of the returned path,
            if any. The function must accept two arguments, the first will be an
            integer, the position of the path element, and the second a string, the path
            element itself. Indices/positions start at 0 for the first element.
        :return: A possibly empty list of nonterminals connecting the source and target
            nonterminals in the graph. Any nonempty returned list begins with
            :code:`source_nonterminal` and ends with :code:`target_nonterminal`. If
            source and target nonterminals are identical and there is a connection from
            that nonterminal to itself, the returned list contains one element, which
            is that nonterminal.
        """

        source_nodes = self.nodes(source_symbol)
        target_nodes = self.nodes(target_symbol)

        if source_nodes is None or target_nodes is None:
            return None

        paths = [
            self.shortest_path_between_nodes(source_node, target_node, node_filter)
            for source_node in source_nodes
            for target_node in target_nodes
        ]

        paths = [path for path in paths if path and len(path) > 1]

        if not paths:
            return []

        return sorted(paths, key=len)[0]

    def shortest_path_between_nodes(
        self,
        source_node: Node,
        target_node: Node,
        node_filter: Callable[[Node], bool] = lambda node: not isinstance(
            node, ChoiceNode
        ),
    ) -> Optional[List[Node]]:
        """
        Computest the shortest path between two nodes in the grammar graph
        using a breadth-first search. Note that a more general algorithm like
        Dijkstra's is not required here since all edge weights in grammar graphs
        are equal.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The shortest path between :code:`<stmt>` and :code:`<digit>` in the above
        grammar consists of four nonterminals:

        >>> print(path_to_str(graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<stmt>"), NonterminalNode(0, "<digit>"))))
        <stmt> (0), <assgn> (0), <rhs> (0), <digit> (0)

        If we relax the default :code:`node_filter`, we are also presented the
        "choice nodes:"

        >>> print(path_to_str(graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<stmt>"),
        ...     NonterminalNode(0, "<digit>"),
        ...     lambda node: True)))
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        For unconnected nonterminals (grammar graphs are directed!), we obtain an empty
        list.

        >>> graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<digit>"), NonterminalNode(0, "<stmt>"))
        []

        If a nonterminal is reachable from itself, the algorithm returns a list of
        length one. Call :meth:`shortest_non_trivial_path` for a path of length > 1
        starting and ending in :code:`source_nonterminal`.

        >>> graph.shortest_path_between_nodes(
        ...     NonterminalNode(1, "<stmt>"), NonterminalNode(1, "<stmt>"))
        [NonterminalNode(ident=1, value='<stmt>')]

        Note that there might be multiple nodes of a given nonterminal symbol in the
        graph (occurring in different contexts). The result depends on which one you
        choose.

        >>> graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<stmt>"), NonterminalNode(0, "<stmt>"))
        []

        Reachability is not reflexive: If there is no path of at least one edge between
        a nonterminal and itself, an empty list is returned.

        >>> graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<digit>"), NonterminalNode(0, "<digit>"))
        []

        If either the source or the target symbol does not exist, :code:`None` is
        returned.
        >>> graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<single-digit>"),
        ...     NonterminalNode(0, "<digit>")) is None
        True

        Nodes in the result can be arbitrarily filtered; let's take out the source
        nonterminal:

        >>> print(path_to_str(graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<stmt>"),
        ...     NonterminalNode(0, "<digit>"),
        ...     lambda node: (
        ...         not isinstance(node, NonterminalNode) or
        ...         node.value != "<stmt>"))))
        <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        :param source_node: The start node for the computation of a shortest path.
        :param target_node: The destination node for the computation of a shortest path.
        :param node_filter: A function for filtering out elements of the returned path,
            if any. The function must accept two arguments, the first will be an
            integer, the position of the path element, and the second a string, the path
            element itself. Indices/positions start at 0 for the first element.
        :return: A possibly empty list of nonterminals connecting the source and target
            nonterminals in the graph. Any nonempty returned list begins with
            :code:`source_nonterminal` and ends with :code:`target_nonterminal`. If
            source and target nonterminals are identical and there is a connection from
            that nonterminal to itself, the returned list contains one element, which
            is that nonterminal.
        """

        result = self.__shortest_path_between_nodes(source_node, target_node)
        if result is None:
            return None

        return [node for node in result if node_filter(node)]

    @cache
    def __shortest_path_between_nodes(
        self,
        source_node: Node,
        target_node: Node,
    ) -> Optional[List[Node]]:
        """
        Like :meth:`neo_grammar_graph.NeoGrammarGraph.shortest_path_between_nodes`,
        but without a :code:`node_filter`. Used for caching.

        :param source_node: The start node for the computation of a shortest path.
        :param target_node: The destination node for the computation of a shortest path.
        :return: A possibly empty list of nonterminals connecting the source and target
            nonterminals in the graph. Any nonempty returned list begins with
            :code:`source_nonterminal` and ends with :code:`target_nonterminal`. If
            source and target nonterminals are identical and there is a connection from
            that nonterminal to itself, the returned list contains one element, which
            is that nonterminal.
        """

        source_vertex = self.vertex_to_node.inverse.get(source_node, None)
        target_vertex = self.vertex_to_node.inverse.get(target_node, None)

        if source_vertex is None or target_vertex is None:
            return None

        pred: Dict[Vertex, Vertex] = {}

        class ShortestPathVisitor(BFSVisitor):
            def examine_edge(self, e: Edge):
                if e.target() not in pred:
                    pred[e.target()] = e.source()
                    if e.target() == target_vertex:
                        raise StopSearch()

        bfs_search(
            self.graph,
            source_vertex,
            ShortestPathVisitor(),
        )

        if target_vertex not in pred:
            # Nonterminal is unreachable
            return []

        result: List[Node] = [target_node]
        current_vertex = target_vertex
        while current_vertex != source_vertex:
            pred_vertex = pred[current_vertex]
            result.insert(0, self.vertex_to_node[pred_vertex])
            current_vertex = pred_vertex

        return result

    def paths_between(
        self,
        start_node: str | Node,
        target_node: str | Node,
        node_filter: Callable[[Node], bool] = lambda node: not isinstance(
            node, ChoiceNode
        ),
    ) -> Optional[OrderedSet[Tuple[Node, ...]]]:
        r"""
        Returns a list of all paths between the given grammar symbols.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        There are three paths from :code:`<stmt>` to :code:`digit`:

        >>> print(paths_to_str(graph.paths_between("<stmt>", "<digit>")))
        <stmt> (0), <assgn> (0), <rhs> (0), <digit> (0)
        <stmt> (0), <stmt> (1), <assgn> (1), <rhs> (0), <digit> (0)
        <stmt> (0), <assgn> (1), <rhs> (0), <digit> (0)

        Per default, we only look up paths from the first node corresponding to a
        :code:`start_node` passed as string. However, this method also accepts
        Node objects for one or both source/target arguments:

        >>> print(paths_to_str(graph.paths_between(NonterminalNode(1, "<stmt>"), "<digit>")))
        <stmt> (1), <assgn> (0), <rhs> (0), <digit> (0)
        <stmt> (1), <assgn> (1), <rhs> (0), <digit> (0)

        Note that although :code:`<stmt> (1)` has the same children as
        :code:`<stmt> (0)`, the results are different (as in the below example),
        since :code:`<stmt> (1), <stmt> (1)` would not be a *path* (it has repetitions
        of one edge).

        There is no path from :code:`<digit>` to itself:

        >>> str(graph.paths_between("<digit>", "<digit>"))
        '{}'

        Path computation for self-reachable symbols works as expected.

        >>> print(paths_to_str(graph.paths_between("<stmt>", "<stmt>", lambda _: True)))
        <stmt> (0), <stmt>-choice (0), <stmt> (1)

        If a symbol does not exist, we obtain :code:`None`:

        >>> graph.paths_between("<some-digit>", "<digit>") is None
        True

        :param start_node: The start symbol of the desired paths.
        :param target_node: The target symbol of the desired paths.
        :param node_filter: A function for filtering out elements of the returned paths,
            if any. The function must accept two arguments, the first will be an
            integer, the position of the path element, and the second a string, the path
            element itself. Indices/positions start at 0 for the first element.
        :return: A list of all paths between the given grammar symbols or :code:`None`
            if start or target symbols do not exist in the grammar graph.
        """

        start_nodes = (
            [start_node] if isinstance(start_node, Node) else self.nodes(start_node)
        )
        target_nodes = (
            [target_node] if isinstance(target_node, Node) else self.nodes(target_node)
        )

        if start_nodes is None or target_nodes is None:
            return None

        assert not isinstance(start_node, str) or start_nodes[0].ident == 0
        start_vertex = self.vertex_to_node.inverse[start_nodes[0]]
        target_vertices = [self.vertex_to_node.inverse[node] for node in target_nodes]

        return OrderedSet(
            [
                tuple(
                    [
                        self.vertex_to_node[self.graph.vertex(vid)]
                        for vid in path
                        if node_filter(self.vertex_to_node[self.graph.vertex(vid)])
                    ]
                )
                for target_vertex in target_vertices
                for path in all_paths(self.graph, start_vertex, target_vertex)
            ]
        )

    @cache
    def k_paths(
        self,
        k: int,
        up_to: bool = False,
        start_nodes: Optional[Tuple[str | Node, ...]] = None,
        include_terminals=True,
        graph: Optional[Graph] = None,
    ) -> OrderedSet[Tuple[Node, ...]]:
        r"""
        This function computes all paths of length k in the graph, optionally restricted
        to those reachable from the nonterminal :code:`start_nonterminal`. If up_to
        is True, also smaller paths are considered. "Choice nodes" are not considered in
        the length computation, but are included in the returned results. k-paths ending
        in terminal symbols are included if, and only if, include_nonterminals is True.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The 1-paths in the graph correspond to the nonterminal symbols (dictionary
        keys) in the underlying grammar in different contexts.

        >>> print(paths_to_str(graph.k_paths(1, include_terminals=False)))
        <start> (0)
        <stmt> (0)
        <assgn> (0)
        <stmt> (1)
        <assgn> (1)
        <var> (0)
        <rhs> (0)
        <var> (1)
        <digit> (0)

        We can ask for paths starting at a particular nonterminal.

        >>> print(paths_to_str(graph.k_paths(2, start_nodes=("<digit>",))))
        <digit> (0), <digit>-choice (0), '0' (0)
        <digit> (0), <digit>-choice (1), '1' (0)
        <digit> (0), <digit>-choice (2), '2' (0)
        <digit> (0), <digit>-choice (3), '3' (0)
        <digit> (0), <digit>-choice (4), '4' (0)
        <digit> (0), <digit>-choice (5), '5' (0)
        <digit> (0), <digit>-choice (6), '6' (0)
        <digit> (0), <digit>-choice (7), '7' (0)
        <digit> (0), <digit>-choice (8), '8' (0)
        <digit> (0), <digit>-choice (9), '9' (0)

        Paths ending in terminal symbols can be excluded on demand.

        >>> print(paths_to_str(
        ...     graph.k_paths(
        ...         3,
        ...         start_nodes=("<assgn>",),
        ...         include_terminals=False)))
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        We can also ask for the paths starting from a particular *node* and not all
        nodes (including reference nodes) for a nonterminal:

        >>> print(paths_to_str(
        ...     graph.k_paths(
        ...         3,
        ...         start_nodes=(graph.nodes("<assgn>")[0],),
        ...         include_terminals=False)))
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        If up_to is set to True, we also obtain paths shorter than the set k.

        >>> print(paths_to_str(
        ...     graph.k_paths(
        ...         3,
        ...         start_nodes=(graph.nodes("<assgn>")[0],),
        ...         up_to=True,
        ...         include_terminals=False)))
        <assgn> (0)
        <assgn> (0), <assgn>-choice (0), <var> (0)
        <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        For certain configurations, we might obtain an empty set of paths.

        >>> print(graph.k_paths(
        ...     4,
        ...     start_nodes=("<assgn>",),
        ...     include_terminals=False))
        {}

        If the optionally passed graph-tool graph is given, it must have a vertex
        property called "node" pointing to the Node object associated to a given
        vertex.

        :param k: The length of the paths to return. Maximal length if up_to is True,
            otherwise the exact lenght.
        :param up_to: Set to True iff you are interested also in paths shorter than k.
        :param start_nodes: If present, only k-paths in the part of the grammar
            reachable from these nodes/nonterminals will be considered.
        :param include_terminals: Set to True iff you are interested in paths ending
            in terminal symbols.
        :param graph: An optional Graph object if the paths in another graph (such as
            a ParseTree object) should be computed. Must have a vertex property "node"
            pointing to the Node objects associated to vertices. We assume that the
            vertex with index 0 is the start/root vertex of the graph.
        :return: The set of paths according to the chosen parameters.
        """
        if graph is None:
            graph = self.graph
        else:
            assert (
                "node" in graph.vp
            ), "The `node` property is mandatory if you supply a custom Graph object."

        # Each path of k terminal/nonterminal nodes includes k-1 choice nodes
        k += k - 1

        if "node" in graph.vp:

            def vertex_to_node(vertex: int | Vertex) -> Node:
                node = graph.vp.node[vertex]
                assert isinstance(node, Node)
                return node

        else:

            def vertex_to_node(vertex: Vertex) -> Node:
                return self.vertex_to_node[vertex]

        if start_nodes is None:
            start_vertices = tuple(
                vertex
                for vertex in graph.vertices()
                if isinstance(vertex_to_node(vertex), NonterminalNode)
            )

        else:
            start_vertices = tuple(
                node for elem in start_nodes for node in self.vertices(elem)
            )

        result_vertex_paths = k_paths_in_graph(
            graph,
            vertex_to_node,
            k,
            start_vertices,
            up_to,
            include_terminals,
            hash(self),
        )

        return OrderedSet(
            [
                tuple([vertex_to_node(vertex) for vertex in vertex_path])
                for vertex_path in result_vertex_paths
            ]
        )

    def parse_tree_to_graph(self, tree: ParseTree) -> Graph:
        r"""
        Converts a ParseTree to a graph-tool Graph object.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The following ParseTree can be derived from the above grammar:

        >>> tree: ParseTree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<assgn>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', [('<digit>', [('1', [])])])]),
        ...        (' ; ', []),
        ...        ('<stmt>',
        ...         [('<assgn>',
        ...           [('<var>', [('y', [])]),
        ...            (' := ', []),
        ...            ('<rhs>', [('<var>', [('x', [])])])])])])])

        We convert it to a graph...

        >>> tree_graph = graph.parse_tree_to_graph(tree)

        And assert that it is a tree.

        >>> class TreeVisitor(BFSVisitor):
        ...     def __init__(self):
        ...         self.result = True
        ...
        ...     def non_tree_edge(self, e):
        ...         self.result = False
        ...         raise StopSearch()
        >>> v = TreeVisitor()
        >>> bfs_search(tree_graph, tree_graph.vertex(0), v)
        >>> v.result
        True

        >>> print("\n".join(map(str, [
        ...     (tree_graph.vp.label[e.source()], tree_graph.vp.label[e.target()])
        ...     for e in tree_graph.edges()
        ... ])))
        ('<start> (0)', '<start>-choice (0)')
        ('<start>-choice (0)', '<stmt> (0)')
        ('<stmt> (0)', '<stmt>-choice (0)')
        ('<stmt>-choice (0)', '<assgn> (0)')
        ('<stmt>-choice (0)', "' ; ' (0)")
        ('<stmt>-choice (0)', '<stmt> (1)')
        ('<assgn> (0)', '<assgn>-choice (0)')
        ('<stmt> (1)', '<stmt>-choice (1)')
        ('<assgn>-choice (0)', '<var> (0)')
        ('<assgn>-choice (0)', "' := ' (0)")
        ('<assgn>-choice (0)', '<rhs> (0)')
        ('<var> (0)', '<var>-choice (23)')
        ('<rhs> (0)', '<rhs>-choice (1)')
        ('<stmt>-choice (1)', '<assgn> (1)')
        ('<assgn> (1)', '<assgn>-choice (0)')
        ('<var>-choice (23)', "'x' (0)")
        ('<rhs>-choice (1)', '<digit> (0)')
        ('<digit> (0)', '<digit>-choice (1)')
        ('<assgn>-choice (0)', '<var> (0)')
        ('<assgn>-choice (0)', "' := ' (0)")
        ('<assgn>-choice (0)', '<rhs> (0)')
        ('<var> (0)', '<var>-choice (24)')
        ('<rhs> (0)', '<rhs>-choice (0)')
        ('<digit>-choice (1)', "'1' (0)")
        ('<var>-choice (24)', "'y' (0)")
        ('<rhs>-choice (0)', '<var> (1)')
        ('<var> (1)', '<var>-choice (23)')
        ('<var>-choice (23)', "'x' (0)")

        Note: When you plan to save this tree graph to a DOT file, you must before
        delete the "node" property. This property connects the tree vertices to
        Node objects in the grammar graph. It is an Object property that graph-tool
        cannot serialize.

        >>> del tree_graph.vp["node"]
        >>> tree_graph.save("/tmp/graph.dot")

        Apart from the "node" property, the produced graph only contains the "label"
        property that we also used in the previous example.

        If a tree does not conform to the grammar, an
        :code:`neo_grammar_graph.InvalidTreeException` is raised.

        >>> tree = ("<start>", [("<salami>", None)])
        >>> tree_graph = graph.parse_tree_to_graph(tree)
        Traceback (most recent call last):
        ...
        neo_grammar_graph.gg.InvalidTreeException

        :param tree: The parse tree to convert into a graph (with choice nodes).
        :return: A Graph object corresponding to a derivation in this grammar graph,
            including the right choice nodes.
        """

        assert tree
        assert is_nonterminal(tree[0])

        graph = Graph()
        graph.vp.node = graph.new_vertex_property("object")
        graph.vp.label = graph.new_vertex_property("string")

        root = graph.add_vertex()

        stack = [(tree, NonterminalNode(0, tree[0]), root)]
        while stack:
            (_, children), node, v = stack.pop(0)
            graph.vp.label[v] = str(node)
            graph.vp.node[v] = node

            if not children:
                continue

            choice_node: Optional[ChoiceNode] = next(
                (
                    cast(ChoiceNode, child)
                    for child in self.children(node)
                    if list(
                        map(
                            lambda symbolic_node: symbolic_node.value,
                            self.children(child),
                        )
                    )
                    == list(map(lambda grandchild: grandchild[0], children))
                ),
                None,
            )

            if choice_node is None:
                raise InvalidTreeException

            choice_node_vertex = graph.add_vertex()
            graph.vp.label[choice_node_vertex] = str(choice_node)
            graph.vp.node[choice_node_vertex] = choice_node
            graph.add_edge(v, choice_node_vertex)

            for tree_child, graph_child in zip(
                children or [], self.children(choice_node) or []
            ):
                child_vertex = graph.add_vertex()
                stack.append((tree_child, graph_child, child_vertex))
                graph.add_edge(choice_node_vertex, child_vertex)

        return graph

    def tree_is_valid(self, tree: ParseTree) -> bool:
        """
        Checks whether the given parse tree is valid with respect to this grammar graph.

        Example:

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        The following tree results from parsing the expression :code:`x := 1` in the
        above grammar.

        >>> tree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<assgn>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', [('<digit>', [('1', [])])])])])])

        Thus, it is a valid tree.

        >>> graph.tree_is_valid(tree)
        True

        If we change one nonterminal to a non-existent one, we obtain an invalid tree:

        >>> tree = (
        ...   '<start>',
        ...     [('<stmt>',
        ...       [('<ASSGN>',
        ...         [('<var>', [('x', [])]),
        ...          (' := ', []),
        ...          ('<rhs>', [('<digit>', [('1', [])])])])])])

        >>> graph.tree_is_valid(tree)
        False

        Generally, the root of the derivation tree must be :code:`<start>` for this
        method to return True. If you want to validate a sub tree, you must before
        extract the appropriate subgrammar graph and, if needed, add a new
        :code:`<start>` nonterminal root.

        >>> graph.tree_is_valid(("<asdf>", None))
        False

        :param tree: The tree the validity of which should be evaluated.
        :return: True iff the given tree is valid w.r.t. this grammar graph (note:
            not a subgraph of the grammar graph).
        """

        if tree[0] != "<start>":
            return False

        try:
            self.parse_tree_to_graph(tree)
            return True
        except InvalidTreeException:
            return False

    def save_to_dot(self, file_name: str) -> None:
        """
        Saves the graph as a DOT digraph that can be, e.g., exported to a PNG file
        using :code:`dot -Tpng dot_file_name.dot -o out.png`. If the given file name
        does not end in :code:`.dot`, this ending is appended to the file name.

        >>> import string
        >>> grammar = {
        ...     "<start>":
        ...         ["<stmt>"],
        ...     "<stmt>":
        ...         ["<assgn> ; <stmt>", "<assgn>"],
        ...     "<assgn>":
        ...         ["<var> := <rhs>"],
        ...     "<rhs>":
        ...         ["<var>", "<digit>"],
        ...     "<var>": list(string.ascii_lowercase),
        ...     "<digit>": list(string.digits)
        ... }
        >>> graph = NeoGrammarGraph(grammar)

        >>> graph.save_to_dot("/tmp/grammar-graph.dot")
        >>> import pathlib
        >>> contents = pathlib.Path("/tmp/grammar-graph.dot").read_text()
        >>> print(contents[:123])
        digraph G {
        0 [label="<start> (0)"];
        1 [label="<start>-choice (0)"];
        2 [label="<stmt> (0)"];
        3 [label="<stmt>-choice (0)"];

        >>> graph.save_to_dot("/tmp/no-dot-file.txt")
        >>> pathlib.Path("/tmp/no-dot-file.txt").exists()
        False

        >>> pathlib.Path("/tmp/no-dot-file.txt.dot").exists()
        True

        :param file_name: The path to the file into which the exported DOT code will
            be stored. A :code:`.dot` ending is appended if not already present.
        :return: Nothing. Stores a file as side effect.
        """

        assert not os.path.isdir(file_name)

        if not file_name.endswith(".dot"):
            file_name += ".dot"

        self.graph.save(file_name)


def leaves_in_graph(
    graph: Graph, as_ndarray=False, numpy_type=np.uint64
) -> Tuple[Vertex, ...] | np.ndarray:
    """
    This function computes the leaves in a graph using matrix multiplication.
    If as_ndarray is False, it returns a tuple of Vertex elements for all graph leaves.
    Otherwise, it returns an ndarray of elements of type :code:`numpy_type` representing
    the vertex indices.

    :param graph: The graph to find leaves in.
    :param as_ndarray: If True, a numpy ndarray is returned instead of a tuple of
        Vertex objects.
    :param numpy_type: The numeric type the elements of a returned numpy ndarray
        should be cast to.
    :return: The leaf indices.
    """

    matrix = adjacency(graph)
    v = np.array([1 for _ in range(np.shape(matrix)[0])])
    prod = np.matmul(v, matrix.toarray())

    if as_ndarray:
        return np.asarray(
            tuple(
                int(graph.vertex(i, use_index=False))
                for i, elem in enumerate(prod)
                if elem == [0]
            ),
            dtype=np.uint64,
        )

    return tuple(
        graph.vertex(i, use_index=False) for i, elem in enumerate(prod) if elem == [0]
    )


class InvalidTreeException(Exception):
    """
    Signals that a parse tree does not conform to the grammar.
    """

    pass


def path_to_str(path: Sequence[Node], separator: str = ", ") -> str:
    """
    Converts a path (list of nodes) to a :code:`separator`-separated string.

    >>> path_to_str([
    ...     NonterminalNode(ident=0, value='<stmt>'),
    ...     ChoiceNode(ident=0, parent_nonterminal='<stmt>'),
    ...     NonterminalNode(ident=0, value='<assgn>'),
    ...     ChoiceNode(ident=0, parent_nonterminal='<assgn>'),
    ...     NonterminalNode(ident=0, value='<rhs>'),
    ...     ChoiceNode(ident=1, parent_nonterminal='<rhs>'),
    ...     NonterminalNode(ident=0, value='<digit>')])
    '<stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)'

    :param path: The path to convert.
    :param separator: The separator to insert between nodes.
    :return: A string representation of the path.
    """

    return separator.join(map(str, path))


def paths_to_str(
    paths: Sequence[Sequence[Node]],
    path_separator: str = "\n",
    node_separator: str = ", ",
) -> str:
    """
    Converts a list of paths (a path is a list of nodes) to a string.

    >>> print(paths_to_str([
    ...     (
    ...         NonterminalNode(ident=0, value='<start>'),
    ...         ChoiceNode(ident=0, parent_nonterminal='<start>'),
    ...         NonterminalNode(ident=0, value='<stmt>'),
    ...         ChoiceNode(ident=0, parent_nonterminal='<stmt>'),
    ...         NonterminalNode(ident=0, value='<assgn>'),
    ...     ),
    ...     (
    ...         NonterminalNode(ident=0, value='<start>'),
    ...         ChoiceNode(ident=0, parent_nonterminal='<start>'),
    ...         NonterminalNode(ident=0, value='<stmt>'),
    ...         ChoiceNode(ident=0, parent_nonterminal='<stmt>'),
    ...         TerminalNode(ident=0, value=' ; '),
    ...     ),
    ...     (
    ...         NonterminalNode(ident=0, value='<start>'),
    ...         ChoiceNode(ident=0, parent_nonterminal='<start>'),
    ...         NonterminalNode(ident=0, value='<stmt>'),
    ...         ChoiceNode(ident=0, parent_nonterminal='<stmt>'),
    ...         NonterminalNode(ident=1, value='<stmt>'),
    ...     )
    ... ]))
    <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
    <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), ' ; ' (0)
    <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)

    :param paths: The paths to convert to a string.
    :param path_separator: The separator to insert between paths.
    :param node_separator: The separator to insert between nodes.
    :return: A string representation of the paths.
    """

    return path_separator.join(
        map(lambda path: path_to_str(path, separator=node_separator), paths)
    )


@cached(
    cache={},
    key=lambda graph, vertex_to_node, k, start_vertices, up_to=False, include_terminals=True, grammar_hash=0: hashkey(
        adjacency(graph).data.tobytes(),
        k,
        start_vertices,
        up_to,
        include_terminals,
        grammar_hash,
    ),
)
def k_paths_in_graph(
    graph: Graph,
    vertex_to_node: Callable[[Vertex], Node],
    k: int,
    start_vertices: Tuple[Vertex, ...] = (),
    up_to: bool = False,
    include_terminals: bool = True,
    grammar_hash: int = 0,
) -> FrozenOrderedSet[Tuple[Vertex, ...]]:
    """
    Computes the k-paths in the given graph, starting at the specified start vertices.

    :param graph: The graph object in which we should compute the k-paths.
    :param vertex_to_node: A function mapping vertices to nodes.
    :param k: The length of the k-paths (incl. choice nodes).
    :param start_vertices: The start vertices. If not provided, we only return paths
        starting from vertex 0.
    :param up_to: If True, also paths with lengths smaller than k are returned.
    :param include_terminals: Paths ending in terminal symbols are returned if, and
        only if, this parameter is True.
    :param grammar_hash: This parameter distinguishes graphs originating from different
        grammars. It is only used for hashing.
    :return: The k-path vertex sequences in the graph, subject to the specified
        parameters.
    """

    if not start_vertices:
        start_vertices = graph.vertex(0)

    result_vertex_paths: FrozenOrderedSet[Tuple[Vertex, ...]] = FrozenOrderedSet()

    for vertex in start_vertices:
        result_for_node: Dict[int, OrderedSet[Tuple[Vertex, ...]]] = {
            1: OrderedSet([(vertex,)])
        }

        for depth in range(k - 1):
            result_for_node[depth + 2] = OrderedSet(
                [
                    path + (child,)
                    for path in result_for_node[depth + 1]
                    for child in graph.get_out_neighbors(path[-1])
                    if include_terminals
                    or not isinstance(vertex_to_node(child), TerminalNode)
                ]
            )

        if up_to:
            result_vertex_paths = FrozenOrderedSet(
                result_vertex_paths
                | FrozenOrderedSet(
                    [
                        path
                        for depth, paths in result_for_node.items()
                        if depth % 2
                        for path in paths
                    ]
                )
            )
        else:
            assert k in result_for_node
            result_vertex_paths = FrozenOrderedSet(
                result_vertex_paths | result_for_node[k]
            )

    return result_vertex_paths
