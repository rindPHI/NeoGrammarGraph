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
from functools import lru_cache
from typing import Dict, List, Callable, Optional, Tuple, Any, cast

from bidict import MutableBidict, bidict
from graph_tool import Graph, Vertex, Edge, GraphView
from graph_tool.search import bfs_search, BFSVisitor, StopSearch
from graph_tool.topology import transitive_closure, all_paths
from orderedset import OrderedSet

from neo_grammar_graph.helpers import split_expansion, is_nonterminal
from neo_grammar_graph.nodes import (
    ChoiceNode,
    Node,
    NonterminalNode,
    TerminalNode,
)
from neo_grammar_graph.type_defs import Grammar, ParseTree


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

        :param grammar: The grammar to construct the
            :class:`~neo_grammar_graph.NeoGrammarGraph` object from. The grammar must
            contain a nonterminal :code:`<start>` that is interpreted as the start
            symbol of the grammar.
        """

        self.grammar: Grammar = grammar
        self.graph: Graph = Graph()
        self.closure: Optional[Graph] = None
        self.vertex_to_node: MutableBidict[Vertex, Node] = bidict({})
        self.symbol_to_vertices: Dict[str, OrderedSet[Vertex]] = {}

        # Cache
        self.__hash: Optional[int] = None

        self.__initialize_graph()

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
                    split_expansion(alternative) or ['']
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

        :return: A has value for this graph.
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

        >>> print(", ".join(map(str, graph.shortest_path("<stmt>", "<digit>"))))
        <stmt> (0), <assgn> (0), <rhs> (0), <digit> (0)

        If we relax the default :code:`node_filter`, we are also presented the
        "choice nodes:"

        >>> print(", ".join(map(str,
        ...     graph.shortest_path("<stmt>", "<digit>", lambda _: True))))
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        For unconnected nonterminals (grammar graphs are directed!), we obtain an empty
        list.

        >>> graph.shortest_path("<digit>", "<stmt>")
        []

        For nonterminals reachable by themselves, the end node might be a different
        one than the start node (though they share the same nonterminal symbol), since
        the nonterminal is reached via a different context.

        >>> print(", ".join(map(str,
        ...     graph.shortest_path("<stmt>", "<stmt>", lambda _: True))))
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

        >>> print(", ".join(map(str, graph.shortest_path(
        ...     "<stmt>",
        ...     "<digit>",
        ...     lambda node: (
        ...         not isinstance(node, NonterminalNode) or
        ...         node.value != "<stmt>")))))
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
        source_node: str | Node,
        target_node: str | Node,
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

        >>> print(", ".join(map(str, graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<stmt>"), NonterminalNode(0, "<digit>")))))
        <stmt> (0), <assgn> (0), <rhs> (0), <digit> (0)

        If we relax the default :code:`node_filter`, we are also presented the
        "choice nodes:"

        >>> print(", ".join(map(str, graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<stmt>"),
        ...     NonterminalNode(0, "<digit>"),
        ...     lambda node: True))))
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

        >>> print(", ".join(map(str, graph.shortest_path_between_nodes(
        ...     NonterminalNode(0, "<stmt>"),
        ...     NonterminalNode(0, "<digit>"),
        ...     lambda node: (
        ...         not isinstance(node, NonterminalNode) or
        ...         node.value != "<stmt>")))))
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

        return [node for node in result if node_filter(node)]

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

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.paths_between("<stmt>", "<digit>"))))
        <stmt> (0), <assgn> (0), <rhs> (0), <digit> (0)
        <stmt> (0), <stmt> (1), <assgn> (1), <rhs> (0), <digit> (0)
        <stmt> (0), <assgn> (1), <rhs> (0), <digit> (0)

        Per default, we only look up paths from the first node corresponding to a
        :code:`start_node` passed as string. However, this method also accepts
        Node objects for one or both source/target arguments:

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.paths_between(NonterminalNode(1, "<stmt>"), "<digit>"))))
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

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.paths_between("<stmt>", "<stmt>", lambda _: True))))
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

    @lru_cache(maxsize=None)
    def k_paths(
        self,
        k: int,
        up_to: bool = False,
        start_node: Optional[str | Node] = None,
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

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths(1, include_terminals=False))))
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

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths(2, start_node="<digit>"))))
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

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths(
        ...         3,
        ...         start_node="<assgn>",
        ...         include_terminals=False))))
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        If up_to is set to True, we also obtain paths shorter than the set k.

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths(
        ...         3,
        ...         start_node="<assgn>",
        ...         up_to=True,
        ...         include_terminals=False))))
        <assgn> (0)
        <assgn> (0), <assgn>-choice (0), <var> (0)
        <var> (0)
        <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <rhs> (0)
        <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <var> (1)
        <rhs> (0), <rhs>-choice (1), <digit> (0)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <digit> (0)

        For certain configurations, we might obtain an empty set of paths.

        >>> print(graph.k_paths(
        ...     4,
        ...     start_node="<assgn>",
        ...     include_terminals=False))
        {}

        If the optionally passed graph-tool graph is given, it must have a vertex
        property called "node" pointing to the Node object associated to a given
        vertex.

        :param k: The length of the paths to return. Maximal length if up_to is True,
            otherwise the exact lenght.
        :param up_to: Set to True iff you are interested also in paths shorter than k.
        :param start_node: If present, only k-paths in the part of the grammar
            reachable from this nonterminal will be considered.
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

        path_map: Dict[Vertex, List[Tuple[Vertex, ...]]] = {}

        class PathVisitor(BFSVisitor):
            def examine_edge(self, e: Edge):
                path_map.setdefault(e.source(), []).append((e.source(),))
                path_map.setdefault(e.target(), []).append((e.source(), e.target()))

                prefixes = [
                    path for path in path_map.get(e.source(), []) if len(path) < k
                ]
                if not prefixes:
                    return

                path_map[e.target()].extend(
                    [prefix + (e.target(),) for prefix in prefixes]
                )

        if start_node is None:
            start_node = NonterminalNode(0, "<start>")
        else:
            start_nodes = self.nodes(start_node)
            assert start_nodes
            start_node = start_nodes[0]

        if "node" in graph.vp:

            def vertex_to_node(vertex: Vertex) -> Node:
                result = graph.vp.node[vertex]
                assert isinstance(result, Node)
                return result

            start_vertex = graph.vertex(0)
            assert vertex_to_node(start_vertex) == start_node
        else:

            def vertex_to_node(vertex: Vertex) -> Node:
                return self.vertex_to_node[vertex]

            start_vertex = self.vertex_to_node.inverse[start_node]

        bfs_search(graph, start_vertex, PathVisitor())

        # We collect all the paths from `path_map` that do not start or end
        # in a choice node and have a suitable length. Furthermore, we eliminate
        # paths ending in terminal symbols if the corresponding flag is set.
        paths = [
            path
            for path_set in path_map.values()
            for path in path_set
            if not isinstance(vertex_to_node(path[0]), ChoiceNode)
            and not isinstance(vertex_to_node(path[-1]), ChoiceNode)
            and (up_to or len(path) == k)
            and (
                include_terminals
                or not isinstance(vertex_to_node(path[-1]), TerminalNode)
            )
        ]

        # For the final result, we convert vertex objects to grammar string symbols.
        return OrderedSet([tuple([vertex_to_node(v) for v in path]) for path in paths])

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

    def k_paths_in_tree(
        self,
        tree: ParseTree,
        k: int,
        include_potential_paths=True,
        include_terminals=True,
    ) -> OrderedSet[Tuple[Node, ...]]:
        r"""
        This method computes the grammar k-paths covered by the given
        :class:`~neo_grammar_graph.type_defs.ParseTree`. The length of
        the paths is specified by the parameter "k." In the value for
        k, choice nodes are not included. That is, for k=3, paths of length
        5 (including two choice nodes) are computed. If :code:`include_potential_paths`
        is True, then k-paths reachable from open leaves in the tree are included.
        If :code:`include_terminals` is True, then paths ending in terminal symbols
        are included.

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

        This is the set of all 3-paths in this tree.

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths_in_tree(tree, 3, include_terminals=True))))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), ' ; ' (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), ' := ' (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <var> (0), <var>-choice (23), 'x' (0)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), ' := ' (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)
        <rhs> (0), <rhs>-choice (1), <digit> (0), <digit>-choice (1), '1' (0)
        <assgn> (1), <assgn>-choice (0), <var> (0), <var>-choice (24), 'y' (0)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <rhs> (0), <rhs>-choice (0), <var> (1), <var>-choice (23), 'x' (0)

        Optionally, we can exclude the paths ending in terminal symbols:

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths_in_tree(
        ...         tree, 3, include_terminals=False))))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)

        Let us "open" a tree leave by replacing a subtree with :code:`None`:

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
        ...           (' := ', []),
        ...           ('<rhs>', None)])])])])

        First, let us inspect the "concrete" paths (without terminal symbols) in this
        cropped tree:

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths_in_tree(
        ...         tree, 3, include_potential_paths=False, include_terminals=False))))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)

        If we additionally ask for the potential paths, the two paths starting from the
        :code:`<assgn> (1)` node and passing the open :code:`<rhs> (0)` node are added
        to the result:

        >>> print("\n".join(map(
        ...     lambda path: ", ".join(map(str, path)),
        ...     graph.k_paths_in_tree(
        ...         tree, 3, include_potential_paths=True, include_terminals=False))))
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <assgn> (0)
        <start> (0), <start>-choice (0), <stmt> (0), <stmt>-choice (0), <stmt> (1)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <var> (0)
        <stmt> (0), <stmt>-choice (0), <assgn> (0), <assgn>-choice (0), <rhs> (0)
        <stmt> (0), <stmt>-choice (0), <stmt> (1), <stmt>-choice (1), <assgn> (1)
        <assgn> (0), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <var> (0)
        <stmt> (1), <stmt>-choice (1), <assgn> (1), <assgn>-choice (0), <rhs> (0)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (0), <var> (1)
        <assgn> (1), <assgn>-choice (0), <rhs> (0), <rhs>-choice (1), <digit> (0)

        If we take the most "general" tree consisting only of an open root with the
        start nonterminal, the potential k-paths with terminal symbols are the same
        as all k-paths in the grammar graph.

        >>> graph.k_paths_in_tree(("<start>", None), 3) == graph.k_paths(3)
        True

        >>> graph.k_paths_in_tree(("<start>", None), 4) == graph.k_paths(4)
        True

        :param tree: The :class:`~neo_grammar_graph.type_defs.ParseTree` from which to
            extract the k-paths.
        :param k: The length of the paths.
        :param include_potential_paths: Set to True if you want to include potential
            paths that might be reachable for open tree leaves.
        :param include_terminals: Set to True if you want to include terminal symbols
            in the returned k-paths.
        :return: The grammar k-paths covered by the given tree.
        """

        tree_graph = self.parse_tree_to_graph(tree)

        tree_k_paths = self.k_paths(
            k,
            graph=tree_graph,
            up_to=False,
            start_node="<start>",
            include_terminals=include_terminals,
        )

        if not include_potential_paths:
            return tree_k_paths

        result = OrderedSet(tree_k_paths)

        # Add to result:
        #
        # - All k-paths that start with a nonterminal reachable from a nonterminal
        #   of some open tree leaf.
        # - All grammar k-paths for which there is a non-empty suffix of a tree k-path
        #   ending in a nonterminal that is a prefix of the grammar k-path.

        open_leaf_vertices = [
            v
            for v in tree_graph.vertices()
            if bool(next(v.out_neighbors(), False)) is False
            if is_nonterminal(tree_graph.vp.node[v].value)
        ]

        open_leaf_nodes = [tree_graph.vp.node[v] for v in open_leaf_vertices]

        grammar_k_paths = self.k_paths(
            k,
            up_to=False,
            start_node="<start>",
            include_terminals=include_terminals,
        )

        for other_path in grammar_k_paths.difference(set(result)):
            if any(
                other_path[0] == leaf_node or self.reachable(leaf_node, other_path[0])
                for leaf_node in open_leaf_nodes
            ):
                result.add(other_path)

        tree_paths_to_open_leaves = OrderedSet(
            [
                tuple([tree_graph.vp.node[v] for v in path])
                for open_leaf_vertex in open_leaf_vertices
                for path in all_paths(
                    tree_graph, tree_graph.vertex(0), open_leaf_vertex
                )
            ]
        )

        # Add other_path if there is a non-empty suffix of a tree path ending in
        # a nonterminal that is a prefix of other_path.
        #
        # Example:
        #
        # - other_path (length 2*k-1):            1234567
        # - tree_path (length arbitrary): 9999999912345
        # - Shared pre/postfix: 12345 (for shift of 2)
        #
        # `shift` can proceed in steps of 2, since we do not consider paths starting
        # or ending in choice nodes (every second node is a choice node).
        for other_path in grammar_k_paths.difference(set(result)):
            for tree_path in tree_paths_to_open_leaves:
                for shift in range(1, 2 * k - 1, 2):
                    if tree_path[-shift:] == other_path[:shift]:
                        result.add(other_path)

        return result

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


class InvalidTreeException(Exception):
    """
    Signals that a parse tree does not conform to the grammar.
    """

    pass
